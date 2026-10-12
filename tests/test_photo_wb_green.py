"""The green-magenta white-balance axis (``white_balance_g`` / ``_g_high``).

White balance had only half its meaning: an amber-blue axis (r/b) and no
green-magenta one, so a fluorescent or forest-canopy cast was inexpressible —
not by white balance, not by split-tone, not by HSL. These pin the new axis
inside ``_apply_colorbalance``, its reach from a look and from the CLI, and —
the load-bearing half — that the VIDEO grade and its baked LUT do not move: the
video producers never set green, so the neutrality guard still early-returns and
the ffmpeg colorbalance mirror still pins ``gs``/``gm``/``gh`` at 0.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.color_filters import build_filter_chain
from gomoveshift.pipeline.photos import develop as dev

# A representative VIDEO grade and the sha256 of its 17³ baked .cube, captured
# on the pre-green source. This is the bit-identity pin: green rides inside the
# same ``_apply_colorbalance`` the cube baker calls, so a default that failed to
# stay neutral would move every baked LUT — and this hash.
VIDEO_PARAMS = cc.CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=6.0,
    saturation_mult=1.08,
    contrast_curve=0.06,
    white_balance_r=-0.03,
    white_balance_b=0.02,
    white_balance_r_high=-0.02,
    white_balance_b_high=0.015,
    gamma=1.05,
)
# A BAKED LUT CUBE's digest, not a credential: it pins that a video-mode correction
# still bakes byte-for-byte as it did before the green axis existed. detect-secrets
# reads any 64-char hex literal as a high-entropy string, so it is allowlisted here.
VIDEO_CUBE_SHA256 = (
    "c4e0d3450857636c334301930f3266b54bbdbc8f7503f01bcbf0377d49c3c943"  # pragma: allowlist secret
)


def _ramp() -> np.ndarray:
    return np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], 3, axis=1)


# The midtone band the shadow+midtone shift actually reaches (the weights fall to
# zero at 0.25 and again at 0.75), so "it moved" is asserted where it must move.
def _mids(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, 0] > 0.3) & (rgb[:, 0] < 0.7)


def test_a_positive_green_lifts_only_the_green_channel():
    rgb = _ramp()
    out = cc._apply_colorbalance(rgb, cc.CorrectionParams(white_balance_g=0.05))
    mids = _mids(rgb)

    assert np.array_equal(out[:, 0], rgb[:, 0]), "red moved on a green-only shift"
    assert np.array_equal(out[:, 2], rgb[:, 2]), "blue moved on a green-only shift"
    assert (out[mids, 1] > rgb[mids, 1]).all(), "green did not move"


def test_a_negative_green_pushes_the_other_way_toward_magenta():
    rgb = _ramp()
    out = cc._apply_colorbalance(rgb, cc.CorrectionParams(white_balance_g=-0.05))
    mids = _mids(rgb)

    assert (out[mids, 1] < rgb[mids, 1]).all(), "magenta side did not move green down"
    assert np.array_equal(out[:, [0, 2]], rgb[:, [0, 2]]), "the amber-blue axis moved"


def test_the_high_field_moves_the_highlights_and_leaves_the_shadows():
    """``white_balance_g_high`` carries a cast concentrated in the bright end,
    exactly as its r/b siblings do — the highlight weight is 0 below 0.75."""
    rgb = _ramp()
    out = cc._apply_colorbalance(rgb, cc.CorrectionParams(white_balance_g_high=0.04))

    dark, bright = rgb[:, 0] <= 0.75, rgb[:, 0] >= 0.95
    assert np.array_equal(out[dark, 1], rgb[dark, 1]), "the shadows took a highlight shift"
    assert (out[bright, 1] > rgb[bright, 1]).all(), "the highlights did not move"


def test_a_neutral_green_is_a_bit_identical_no_op():
    """Both the whole-filter early return and the per-channel skip: a params with
    green at its 0.0 default must produce the byte-for-byte pre-green output."""
    rgb = _ramp()

    assert np.array_equal(cc._apply_colorbalance(rgb, cc.CorrectionParams()), rgb)
    warm = cc.CorrectionParams(white_balance_r=0.04, white_balance_b=-0.03)
    out = cc._apply_colorbalance(rgb, warm)
    assert np.array_equal(out[:, 1], rgb[:, 1]), "a neutral green still touched the channel"


def test_the_video_grade_never_sets_green_and_its_baked_cube_is_unchanged(tmp_path):
    """The hard constraint. ``compute_correction`` produces no green in either
    mode, the baked cube still hashes to its pre-green value, and the ffmpeg
    colorbalance mirror the encoder runs still pins green at 0."""
    frames = [
        np.stack(
            [
                rng.integers(35, 60, size=(48, 64)),
                rng.integers(18, 34, size=(48, 64)),
                rng.integers(6, 16, size=(48, 64)),
            ],
            axis=-1,
        ).astype(np.uint8)
        for rng in [np.random.default_rng(s) for s in range(8)]
    ]
    from gomoveshift.pipeline.color import analyze_segments

    segments, _ = analyze_segments(frames)
    for segment in segments:
        for mode in (cc.GRADE_MODE_VIDEO, cc.GRADE_MODE_PHOTO):
            params = cc.compute_correction(segment, mode=mode)
            assert params.white_balance_g == 0.0 and params.white_balance_g_high == 0.0

    cube = cc.params_to_lut3d_cube(VIDEO_PARAMS, cube_size=17, output_path=tmp_path / "v.cube")
    assert hashlib.sha256(cube.read_bytes()).hexdigest() == VIDEO_CUBE_SHA256

    chain = build_filter_chain([VIDEO_PARAMS])
    assert "gs=0:" in chain and "gm=0:" in chain and "gh=0:" in chain


def test_the_knob_reaches_a_look_and_a_styled_variant(tmp_path):
    """Named in a look it lands on that look's own record, and through the shared
    field name it flows into the colour chain ``_look_params`` lays on."""
    resolved = dev.look_variant_options({"white_balance_g": 0.3}, dev.DevelopOptions())
    assert resolved.white_balance_g == 0.3
    assert dev._look_params(cc.CorrectionParams(), resolved).white_balance_g == 0.3

    assert PhotosConfig(looks={"tint": {"white_balance_g": 0.3}}).looks
    with pytest.raises(ValidationError):
        PhotosConfig(looks={"tint": {"white_balance_g": 0.9}})

    photo = _write_photo(tmp_path / "IMG_0201.jpg")
    plan = dev.plan_variants(photo, ["natural"], options=dev.DevelopOptions(white_balance_g=0.2))
    assert [p.white_balance_g for _, p in plan.styled] == [0.2]
    assert plan.base.white_balance_g == 0.0, "an operator knob must never ride the base grade"


def _write_photo(path: Path) -> Path:
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv: str) -> int:
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_the_cli_flag_threads_through_and_out_of_range_is_refused_pre_decode(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(developed_output_dir=str(tmp_path / "out"))),
    )
    photo = _write_photo(tmp_path / "IMG_0202.jpg")
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --white-balance-g 0.4") == 0
    assert [o.white_balance_g for o in seen] == [0.4]

    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad green shift"))
    assert _run(f"photos develop {photo} --white-balance-g 0.9") == 1
    assert "--white-balance-g must be in [-0.5, 0.5]" in capsys.readouterr().err


def test_the_green_axis_feeds_the_develop_cache_version():
    """The applier's source is hashed, so a green retune re-develops cached pixels."""
    assert cc._apply_colorbalance in dev._HASHED_SOURCES
