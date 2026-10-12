"""External ``.cube`` 3D LUT reader + NumPy applier (film-emulation looks).

The reader/applier itself now ships as the published ``pycube-lut`` package — this
service holds no second copy. What is pinned here is the SEAM: that the package the
photo path imports still behaves the way this pipeline's grade engine, its
``SceneProfile`` and its style constants assume.
"""

from typing import get_type_hints

import numpy as np
import pytest
from pycube_lut import (
    LUT_1D_TO_3D_SIZE,
    Cube,
    CubeError,
    Scene,
    apply_cube,
    load_cube,
    score_lut,
)
from pycube_lut.select import WARM_FRACTION_ALREADY_WARM

from gomoveshift.pipeline.color_correction import CorrectionParams, params_to_lut3d_cube
from gomoveshift.pipeline.photos.develop import _grade
from gomoveshift.pipeline.photos.scene import SceneProfile
from gomoveshift.pipeline.photos.styles import GOLDEN_WARM_FRACTION_HIGH


def _write_cube(path, size, fn) -> None:
    """Write a size^3 .cube whose grid sample at (r,g,b) is ``fn(r,g,b)`` — in the
    Adobe R-fastest order the reader must honor."""
    axis = np.linspace(0.0, 1.0, size)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for b in axis:
        for g in axis:
            for r in axis:
                o = fn(r, g, b)
                lines.append(f"{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_load_cube_parses_size_domain_and_indexing(tmp_path):
    p = tmp_path / "id.cube"
    _write_cube(p, 3, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    assert cube.size == 3
    assert np.allclose(cube.domain_min, 0.0) and np.allclose(cube.domain_max, 1.0)
    # table is indexed [r, g, b]: the identity grid must read back as its coords.
    assert np.allclose(cube.table[2, 0, 0], [1.0, 0.0, 0.0])  # max R, no G/B
    assert np.allclose(cube.table[0, 2, 0], [0.0, 1.0, 0.0])  # max G


def test_identity_lut_is_a_near_noop(tmp_path):
    p = tmp_path / "id.cube"
    _write_cube(p, 33, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    out = apply_cube(img, cube)
    # Trilinear interpolation of an identity (affine) grid is exact bar rounding.
    assert np.abs(out.astype(int) - img).max() <= 1


def test_channel_swap_lut_reorders_channels(tmp_path):
    # A LUT that outputs (b, r, g) proves the applier honors per-channel indexing.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 200  # pure-ish red in
    out = apply_cube(img, cube)
    # red input -> the LUT routes R into the G output channel
    assert out[0, 0, 1] > 190 and out[0, 0, 0] < 10 and out[0, 0, 2] < 10


@pytest.mark.parametrize(
    "params",
    [
        CorrectionParams(saturation_mult=1.3, contrast_curve=0.08, black_point=-6.0),
        CorrectionParams(shadow_lift=10.0, gamma=1.15, white_balance_r=0.04, black_point=8.0),
    ],
)
def test_baked_lut_matches_the_direct_grade(tmp_path, params):
    # Round-trip parity: the engine bakes params -> .cube (video path); the reader
    # + applier must reproduce the SAME numpy grade the photo path renders directly.
    cube_path = tmp_path / "baked.cube"
    params_to_lut3d_cube(params, cube_size=64, output_path=cube_path)
    cube = load_cube(cube_path)

    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    via_lut = apply_cube(img, cube).astype(int)
    via_grade = _grade(img, params).astype(int)
    assert np.abs(via_lut - via_grade).mean() < 1.5  # cube quantization + trilinear only


def test_domain_scaling_is_honored(tmp_path):
    # A LUT declaring a 0..0.5 domain maps mid-grey to the grid's top.
    p = tmp_path / "dom.cube"
    axis = np.linspace(0.0, 1.0, 2)
    lines = ["LUT_3D_SIZE 2", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 0.5 0.5 0.5"]
    for b in axis:
        for g in axis:
            for r in axis:
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    p.write_text("\n".join(lines) + "\n")
    cube = load_cube(p)
    img = np.full((2, 2, 3), 128, dtype=np.uint8)  # ~0.5 in -> domain max -> grid 1.0
    out = apply_cube(img, cube)
    assert out.min() >= 250


def test_comments_and_title_are_skipped(tmp_path):
    p = tmp_path / "c.cube"
    body = "\n".join(f"{v:.3f} {v:.3f} {v:.3f}" for v in (0.0, 1.0) for _ in range(4))
    p.write_text(f'# a comment\nTITLE "Film Look"\nLUT_3D_SIZE 2\n\n{body}\n')
    cube = load_cube(p)
    assert cube.size == 2


def test_wrong_row_count_raises(tmp_path):
    p = tmp_path / "short.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")  # needs 8 rows
    with pytest.raises(CubeError, match="expected 8 data rows"):
        load_cube(p)


def _write_hald(path, level, fn) -> None:
    """Write a level-``level`` HALD CLUT PNG (side ``level**3`` px, cube size
    ``level**2``) whose sample at (r,g,b) is ``fn(r,g,b)`` — R-fastest row-major,
    the same order a .cube's data rows use. ``fn`` takes/returns float arrays."""
    from PIL import Image

    size, side = level**2, level**3
    idx = np.arange(side * side)
    coords = (
        (idx % size) / (size - 1),
        ((idx // size) % size) / (size - 1),
        (idx // size**2) / (size - 1),
    )
    out = np.stack(fn(*coords), axis=1)
    arr = (out * 255).round().astype(np.uint8).reshape(side, side, 3)
    Image.fromarray(arr).save(path)


def test_hald_identity_round_trips(tmp_path):
    p = tmp_path / "id.png"
    _write_hald(p, 2, lambda r, g, b: (r, g, b))  # level 2 -> 8x8 px, 4^3 cube
    cube = load_cube(p)
    assert cube.size == 4
    assert np.allclose(cube.domain_min, 0.0) and np.allclose(cube.domain_max, 1.0)
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    assert np.abs(apply_cube(img, cube).astype(int) - img).max() <= 1


def test_hald_channel_swap_reorders_channels(tmp_path):
    p = tmp_path / "swap.png"
    _write_hald(p, 3, lambda r, g, b: (b, r, g))  # level 3 -> 27x27 px, 9^3 cube
    cube = load_cube(p)
    assert cube.size == 9
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 200  # red in -> the LUT routes R into the G output channel
    out = apply_cube(img, cube)
    assert out[0, 0, 1] > 190 and out[0, 0, 0] < 10 and out[0, 0, 2] < 10


@pytest.mark.parametrize(
    "shape, match",
    [((8, 16), "square"), ((5, 5), "perfect cube"), ((1, 1), "at least 2")],
)
def test_hald_bad_geometry_raises(tmp_path, shape, match):
    from PIL import Image

    p = tmp_path / "bad.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(p)
    with pytest.raises(CubeError, match=match) as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_unreadable_hald_png_raises(tmp_path):
    p = tmp_path / "junk.PNG"  # suffix dispatch is case-insensitive
    p.write_bytes(b"not a png at all")
    with pytest.raises(CubeError, match="cannot read HALD CLUT PNG") as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_one_d_lut_becomes_a_separable_3d_cube(tmp_path):
    # Three per-channel tone curves: R inverted, G and B identity.
    n = 16
    axis = np.linspace(0.0, 1.0, n)
    p = tmp_path / "curve.cube"
    p.write_text(f"LUT_1D_SIZE {n}\n" + "".join(f"{1 - v:.6f} {v:.6f} {v:.6f}\n" for v in axis))
    cube = load_cube(p)
    assert cube.size == LUT_1D_TO_3D_SIZE  # resampled, never 1024^3
    out = apply_cube(np.full((2, 2, 3), 64, np.uint8), cube)
    assert abs(int(out[0, 0, 0]) - 191) <= 2  # R inverted
    assert abs(int(out[0, 0, 1]) - 64) <= 2  # G untouched
    assert abs(int(out[0, 0, 2]) - 64) <= 2  # B untouched


@pytest.mark.parametrize(
    "text, match",
    [
        ("LUT_1D_SIZE 4\n0 0 0\n1 1 1\n", "expected 4 data rows"),
        ("LUT_1D_SIZE x\n0 0 0\n", "malformed LUT_1D_SIZE"),
        ("LUT_1D_SIZE 1\n0 0 0\n", "must be >= 2"),
    ],
)
def test_one_d_lut_failures_raise(tmp_path, text, match):
    p = tmp_path / "bad1d.cube"
    p.write_text(text)
    with pytest.raises(CubeError, match=match) as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_three_d_cube_parse_is_unchanged(tmp_path):
    # Pin: the pre-existing 3D path parses byte-identically to an [r,g,b] identity grid.
    p = tmp_path / "id5.cube"
    _write_cube(p, 5, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    expected = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).astype(np.float32)
    assert cube.size == 5
    assert np.array_equal(cube.table, expected)
    assert np.array_equal(cube.domain_min, np.zeros(3, np.float32))
    assert np.array_equal(cube.domain_max, np.ones(3, np.float32))


def test_missing_size_raises(tmp_path):
    p = tmp_path / "nosize.cube"
    p.write_text("0 0 0\n1 1 1\n")
    with pytest.raises(CubeError, match="not a 3D .cube"):
        load_cube(p)


def test_unreadable_file_raises():
    with pytest.raises(CubeError, match="cannot read LUT"):
        load_cube("/no/such/film.cube")


def test_apply_requires_hwc3():
    cube = Cube(
        2, np.zeros((2, 2, 2, 3), np.float32), np.zeros(3, np.float32), np.ones(3, np.float32)
    )
    with pytest.raises(CubeError, match="HxWx3"):
        apply_cube(np.zeros((4, 4), np.uint8), cube)


def test_default_strength_matches_full_strength_byte_for_byte(tmp_path):
    # strength defaults to 1.0 — the shipped full-LUT output, unchanged bit for bit.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.array_equal(apply_cube(img, cube), apply_cube(img, cube, 1.0))


def test_zero_strength_is_a_passthrough_to_the_input(tmp_path):
    # A channel-swap LUT visibly re-grades at full strength; at 0.0 it is a no-op.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.abs(apply_cube(img, cube, 1.0).astype(int) - img).mean() > 20  # full moves pixels
    assert np.abs(apply_cube(img, cube, 0.0).astype(int) - img).max() <= 1  # input, bar rounding


def test_intermediate_strength_lands_between_input_and_full_lut(tmp_path):
    # A LUT that crushes everything to black: half strength is halfway to black.
    p = tmp_path / "black.cube"
    _write_cube(p, 2, lambda r, g, b: (0.0, 0.0, 0.0))
    cube = load_cube(p)
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert apply_cube(img, cube, 1.0).max() <= 1  # full LUT -> black
    assert np.abs(apply_cube(img, cube, 0.5).astype(int) - 100).max() <= 1  # midpoint of 200 and 0


# ── The seam to pycube-lut ──────────────────────────────────────────
# Two couplings the internal copy had by import and the package now only has by
# CONVENTION. Nothing checks either at runtime, so a drift on either side would
# silently mis-score every --lut-auto pick instead of failing. These pin them.


def test_scene_profile_still_satisfies_the_package_scene_protocol():
    """``score_lut`` takes a structural ``Scene``; the photo path hands it a real
    ``SceneProfile``. ``Scene`` is a plain Protocol — no runtime check — so compare
    the two attribute by attribute: a rename, a dropped field or a type change on
    EITHER side fails here rather than at scoring time."""
    required = get_type_hints(Scene)
    supplied = get_type_hints(SceneProfile)
    assert required, "pycube_lut.Scene declares no attributes — the package API moved"
    missing = sorted(name for name in required if name not in supplied)
    assert not missing, f"SceneProfile no longer supplies Scene attributes: {missing}"
    drifted = {n: (required[n], supplied[n]) for n in required if supplied[n] is not required[n]}
    assert not drifted, f"type drift between pycube_lut.Scene and SceneProfile: {drifted}"


def test_a_real_scene_profile_scores_without_adaptation():
    # Structural conformance is necessary but not sufficient — actually score with one.
    cube = Cube(
        2, np.zeros((2, 2, 2, 3), np.float32), np.zeros(3, np.float32), np.ones(3, np.float32)
    )
    scene = SceneProfile("normal", "neutral", "normal", "clear", "normal", False)
    assert isinstance(score_lut(cube, scene, np.full((8, 8, 3), 120, np.uint8)), float)


def _cool_scene(warm_fraction: float) -> SceneProfile:
    # Only a COOL scene reads warm_fraction — it decides whether warmth is still wanted.
    return SceneProfile(
        "normal", "cool", "normal", "clear", "normal", False, warm_fraction=warm_fraction
    )


def test_package_warm_threshold_still_equals_the_service_constant():
    """``score_lut`` used to import ``styles.GOLDEN_WARM_FRACTION_HIGH``; the package
    inlined it. The import is gone, so only this pins the two together."""
    assert WARM_FRACTION_ALREADY_WARM == GOLDEN_WARM_FRACTION_HIGH


def test_the_already_warm_cutoff_bites_at_the_service_constant(tmp_path):
    """The number above, pinned as BEHAVIOUR: a cool frame at the service's threshold
    must already read warm to the package, and just under it must not."""
    p = tmp_path / "id.cube"
    _write_cube(p, 5, lambda r, g, b: (r, g, b))
    cube, frame = load_cube(p), np.full((8, 8, 3), 120, np.uint8)
    at = score_lut(cube, _cool_scene(GOLDEN_WARM_FRACTION_HIGH), frame)
    under = score_lut(cube, _cool_scene(GOLDEN_WARM_FRACTION_HIGH - 0.01), frame)
    assert at != under, "the cutoff no longer lands on the service's constant"
    assert under == score_lut(cube, _cool_scene(0.0), frame)  # everything under is one branch
