"""The MASKED ``--look`` pack: grades that ride the frame's own regions.

``masks.py`` shipped the named soft masks and ``layers:`` wired them into a look,
but ``_MASKED`` was still empty — the engine had nothing in the box to render.
These are that payoff: a sky/ground split, a signage tint, a centre/edge wash and
a bottom-band glow.

What a hermetic test can prove is what the pack promises: every preset is legal on
the REAL config validator (``check_layers`` included), each renders past
``MIN_RENDERED_DISTINCTNESS`` from the natural grade AND from every sibling, each
reproduces exactly, and — the whole point of the pack — each moves the region its
mask names while the complement moves the other way or not at all. Whether a
preset is at the right STRENGTH on a real NEF is a human eyeball check on the reference
card, so every assertion is directional, never a magic pixel value.

Rendered through ``develop_variants`` rather than the CLI door the sibling packs
use, because these assertions are about EXACT complements and the exported JPEG
quantizes in 8x8 blocks — a block straddling a mask boundary would move pixels no
layer touched. ``sharpen=0`` for the same reason: the unsharp mask is a
neighbourhood op and would smear a layered region over its own boundary. A look
carrying layers through the real CLI door is pinned in test_photo_masked_layers.py.
"""

from itertools import combinations, product

import numpy as np
import pytest
from PIL import Image

from gomoveshift.config import PhotosConfig
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos import masks as masks_mod
from gomoveshift.pipeline.photos.develop import MIN_RENDERED_DISTINCTNESS
from gomoveshift.pipeline.photos.looks import _MASKED, BUILTIN_LOOKS

PACK = ("split_world", "neon_edge", "vignette_wash", "golden_ground", "subject_pop")
# Landscape, because ``_sky_mask`` returns all zeros on a portrait frame. The sky
# fills the upper third the mask's band actually meters; the ground below it is
# darker and warm, and ramps left-to-right so the luma masks have a span to
# partition and the highlight window has a real zero region under it.
HEIGHT, WIDTH, HORIZON = 96, 128, 30
# ...and lit signage across that ground, because a landscape whose only bright
# region IS the sky is the frame ``neon_edge`` must NOT grade. Each sign covers
# 1.4% of the frame against the sky's 31% — a 20x separation in AREA at the same
# end of the tonal span, which is the whole reason one mask can keep the signs and
# drop the sky. They are deliberately a little DIMMER than the sky (matching it
# would move the frame's own percentiles and with them every other preset's
# render), so the split under test is size and never brightness. All of them sit
# BELOW the half line, so ``golden_ground``'s untouched top half stays exact.
SIGN_ROWS, SIGN_COLS, SIGN_H, SIGN_W = (54, 76), (3, 24, 45, 66, 87, 108), 12, 14


def _frame(path):
    rgb = np.empty((HEIGHT, WIDTH, 3), np.float32)
    rgb[:HORIZON] = (0.44, 0.56, 0.82)
    rgb[HORIZON:] = (0.36, 0.28, 0.17)
    rgb[HORIZON:] *= np.linspace(0.62, 1.30, WIDTH, dtype=np.float32)[None, :, None]
    for row, col in product(SIGN_ROWS, SIGN_COLS):
        rgb[row : row + SIGN_H, col : col + SIGN_W] = (0.565, 0.52, 0.43)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB").save(path, "JPEG")
    return path


def _subject_frame(path):
    """The same scene with a stand of fine detail over the right half of the
    ground: one region a texture mask must find and one it must not. Saved at a
    quality that keeps the detail — the pack frame's default would eat it."""
    rgb = np.empty((HEIGHT, WIDTH, 3), np.float32)
    rgb[:HORIZON] = (0.44, 0.56, 0.82)
    rgb[HORIZON:] = (0.36, 0.28, 0.17)
    yy, xx = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    detail = 0.13 * np.sign(np.sin(xx / 2.6) * np.sin(yy / 2.6))
    detail[:HORIZON] = 0.0
    detail[:, : WIDTH // 2] = 0.0
    Image.fromarray(
        (np.clip(rgb + detail[..., None], 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
    ).save(path, "JPEG", quality=95)
    return path


def _render(photo, knobs):
    """One preset through the real render seam, as the uint8 frame it exports."""
    options = dev.look_variant_options(knobs, dev.DevelopOptions(sharpen=0.0))
    return dict(dev.develop_variants(photo, ["natural"], options=options))["natural"].rgb


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """The natural grade, every preset, and ``split_world``'s FIRST layer on its
    own, rendered ONCE off one frame — that last one is the counterfactual a
    two-layer look needs: both its layers cover the frame between them, so
    direction alone cannot tell a masked composite from a whole-frame one."""
    photo = _frame(tmp_path_factory.mktemp("masked") / "IMG_0001.jpg")
    sky_only = {"layers": [_MASKED["split_world"]["layers"][0]]}
    return {"natural": _render(photo, {}), "sky_only": _render(photo, sky_only)} | {
        n: _render(photo, _MASKED[n]) for n in PACK
    }


def _pair(rendered, name):
    """(look, natural) signed — uint8 subtraction wraps where a layer darkens."""
    return rendered[name].astype(np.int16), rendered["natural"].astype(np.int16)


def _region(rendered, mask, low, high, **kw):
    """The named mask's strong region and its weak one, metered on the SAME frame
    the layers metered theirs on. Both must be occupied — an empty side would let
    a mean over nothing stand in for a move that never happened."""
    metered = masks_mod.mask_for(mask, rendered["natural"], **kw)
    strong, weak = metered >= high, metered <= low
    assert strong.any() and weak.any(), f"the fixture frame holds no {mask} split"
    return strong, weak


def _warmth(frame, where):
    return float((frame[..., 0] - frame[..., 2])[where].mean())


def _chroma(frame, where):
    return float((frame.max(axis=2) - frame.min(axis=2))[where].mean())


def _distance(a, b):
    """The project's rendered-distinctness measure: mean absolute 8-bit levels."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_every_masked_preset_is_legal_on_the_real_config_validator():
    """A preset config would reject is one the operator cannot override."""
    assert tuple(_MASKED) == PACK, "the pack ships all four, in order"
    assert PhotosConfig(looks=_MASKED).looks == _MASKED
    assert set(PACK) <= set(BUILTIN_LOOKS), "the pack must reach the public table"
    assert all(look["layers"] for look in _MASKED.values()), "a masked look rides a mask"


def test_each_look_renders_apart_from_natural_and_from_every_sibling(rendered):
    """A masked grade moves part of a frame, so it has less of one to be distinct
    on than a whole-frame preset — it still clears the bar both packs ship under."""
    for name in PACK:
        apart = _distance(rendered[name], rendered["natural"])
        assert apart >= MIN_RENDERED_DISTINCTNESS, f"{name} renders {apart:.2f} from natural"
    for a, b in combinations(PACK, 2):
        apart = _distance(rendered[a], rendered[b])
        assert apart >= MIN_RENDERED_DISTINCTNESS, f"{a} renders {apart:.2f} from {b}"


def test_a_masked_look_renders_the_same_pixels_twice(tmp_path):
    """Every mask is metered off the frame itself, so a non-deterministic one
    would make each eyeball calibration unrepeatable."""
    photo = _frame(tmp_path / "IMG_0002.jpg")
    assert np.array_equal(
        _render(photo, _MASKED["split_world"]), _render(photo, _MASKED["split_world"])
    )


def test_split_world_cools_the_sky_and_warms_the_ground(rendered):
    look, base = _pair(rendered, "split_world")
    sky, ground = _region(rendered, "sky", 0.05, 0.8)
    assert _warmth(look, sky) < _warmth(base, sky), "the sky half must go cool"
    assert _warmth(look, ground) > _warmth(base, ground), "the ground half must go warm"
    # Inside the sky the composite must BE the sky layer: the ground recipe stops
    # at the horizon. Ratio floor, not a pixel value — a ground layer that ignored
    # its mask measures ~1.0 here (it reaches the sky as hard as it reaches the
    # ground) where the shipped one measures ~0.02.
    leak = np.abs(look - rendered["sky_only"].astype(np.int16))
    assert leak[sky].mean() < 0.1 * leak[ground].mean(), "the ground layer leaked into the sky"


def test_neon_edge_tints_the_signs_and_never_the_lit_sky(rendered):
    """The recipe's whole justification, and what ``highlights`` could not give
    it: the lit sky measured here is a quarter of the frame, so a per-pixel tone
    mask grades it magenta. ``glints`` groups the bright pixels into regions and
    drops the big one, so the sky is left BYTE-identical while the signs — the
    same end of the span, a twentieth of the area each — take the tint."""
    look, base = _pair(rendered, "neon_edge")
    signs, dark = _region(rendered, "glints", 0.0, 0.5)
    lit_sky, _ = _region(rendered, ["sky", "highlights"], 0.0, 0.5, op="intersect")
    magenta = lambda f, w: float(((f[..., 0] + f[..., 2]) / 2 - f[..., 1])[w].mean())  # noqa: E731
    assert magenta(look, signs) > magenta(base, signs), "the signs must take the signage tint"
    assert np.array_equal(look[lit_sky], base[lit_sky]), "the lit sky took a signage tint"
    assert np.array_equal(look[dark], base[dark]), "a highlight look reached the shadows"


def test_vignette_wash_drains_the_edges_and_warms_the_centre(rendered):
    look, base = _pair(rendered, "vignette_wash")
    centre, edge = _region(rendered, "radial", 0.25, 0.75)
    assert _chroma(look, edge) < _chroma(base, edge), "the edges must wash out"
    assert _warmth(look, centre) > _warmth(base, centre), "the centre must hold warm"
    assert _chroma(look, centre) > _chroma(look, edge), "and keep more colour than the edges"


def test_subject_pop_lifts_the_detailed_region_and_drains_the_smooth_one(tmp_path):
    """The recipe's own justification: it rides DETAIL, so it is not the
    centre/edge ``vignette_wash`` already is wearing a second name. Its own frame
    — the pack's fixture is smooth by design and holds no subject to find."""
    photo = _subject_frame(tmp_path / "IMG_0003.jpg")
    frames = {"natural": _render(photo, {}), "subject_pop": _render(photo, _MASKED["subject_pop"])}
    look, base = _pair(frames, "subject_pop")
    detailed, smooth = _region(frames, "texture", 0.1, 0.7)
    assert _chroma(look, detailed) > _chroma(base, detailed), "the subject must gain colour"
    assert _chroma(look, smooth) < _chroma(base, smooth), "the smooth surround must drain"
    apart = _distance(frames["subject_pop"], _render(photo, _MASKED["vignette_wash"]))
    assert apart >= MIN_RENDERED_DISTINCTNESS, f"subject_pop renders {apart:.2f} from vignette_wash"


def test_golden_ground_warms_the_bottom_band_and_never_the_top_half(rendered):
    """``linear`` at the registry defaults ramps from exactly 0 across the top half
    to 1 at the bottom edge, so the untouched complement here is exact."""
    look, base = _pair(rendered, "golden_ground")
    band, above = _region(rendered, "linear", 0.0, 0.5)
    assert _warmth(look, band) > _warmth(base, band), "the ground band must take the glow"
    assert np.array_equal(look[above], base[above]), "the glow climbed past its gradient"
