"""The amber-blue temperature axes, and a door on every white-balance field.

``CorrectionParams`` renders SIX white-balance fields; exactly one of them
(``white_balance_g``) had a flag, a look key and a ``DevelopOptions`` field, so
the whole blue-amber TEMPERATURE axis — the first slider in every comparable
tool — was reachable only by surrendering the style choice to ``daylight``/
``cloudy``/``shade`` (which is why ``cinestill`` fakes tungsten-in-daylight with
a split-tone pair). These pin the five new doors and the two things that make
them UNLIKE the green axis: an operator value must ADD to the measured cast
correction, and the span is the headroom the style rails leave, not green's ±0.5.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from test_photo_wb_green import VIDEO_CUBE_SHA256, VIDEO_PARAMS  # the green axis' own pin

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import WB_OPERATOR_SPAN, Config, PhotosConfig
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.color_filters import build_filter_chain
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos.styles import STYLE_WB_CAP, STYLES, WB_HARD_CAP, apply_style

#: The five axes this change opens, and a value that moves the picture on each.
NEW_AXES = {
    "white_balance_r": 0.06, "white_balance_b": -0.05, "white_balance_r_high": 0.04,
    "white_balance_b_high": -0.03, "white_balance_g_high": 0.05,
}  # fmt: skip


def _write_photo(path):
    """Warm ground under a blue sky — a frame that MEASURES a real cast, which is
    what makes "added to the measured correction" a checkable claim."""
    y = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None]
    rgb = np.empty((96, 128, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.22 + 0.50 * y, 0.34 + 0.28 * y, 0.78 - 0.42 * y
    rgb += np.random.default_rng(4).normal(0.0, 0.03, rgb.shape).astype(np.float32)
    frame = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(frame, "RGB").save(path, "JPEG", quality=95)
    return path


def _render(photo, **knobs):
    o = dev.DevelopOptions(**knobs)
    return dict(dev.develop_variants(photo, ["moody"], options=o))["moody"].rgb


def test_a_warm_request_moves_a_styled_variants_pixels(tmp_path):
    """THE deliverable: an operator asking for warmth gets warmer pixels on a
    STYLED variant. There was no flag, no field and no look key on the amber-blue
    axis before this, so the request could not even be expressed."""
    photo = _write_photo(tmp_path / "IMG_0301.jpg")
    plain = _render(photo)
    warm = _render(photo, white_balance_r=0.08, white_balance_b=-0.08)
    cool = _render(photo, white_balance_r=-0.08, white_balance_b=0.08)

    assert not np.array_equal(warm, plain), "the warm request rendered as nothing"
    assert warm[..., 0].mean() > plain[..., 0].mean(), "red did not warm"
    assert warm[..., 2].mean() < plain[..., 2].mean(), "blue did not cool"
    assert cool[..., 2].mean() > plain[..., 2].mean(), "the other end of the slider is dead"


def test_the_operator_shift_adds_to_the_measured_cast_and_never_rides_the_base(tmp_path):
    """Unlike green — which no producer ever sets — r/b carry the auto cast
    correction, so a ``replace`` would silently DISCARD it. An operator asking
    for +0.06 of warmth must get "the measured cast PLUS 0.06"."""
    photo = _write_photo(tmp_path / "IMG_0302.jpg")
    plain = dev.plan_variants(photo, ["natural"], options=dev.DevelopOptions())
    assert plain.base.white_balance_b != 0.0, "the fixture must MEASURE a cast to prove this"

    shifted = dev.plan_variants(photo, ["natural"], options=dev.DevelopOptions(**NEW_AXES))
    for axis, asked in NEW_AXES.items():
        before, after = getattr(plain.styled[0][1], axis), getattr(shifted.styled[0][1], axis)
        assert after == pytest.approx(before + asked), f"{axis} replaced the measured cast"
    assert shifted.base == plain.base, "an operator knob must never ride the base grade"


def test_the_span_is_the_headroom_the_style_rails_leave(tmp_path):
    """The rails run inside ``apply_style``, BEFORE the operator value lands, so
    they cannot clamp it — the span is chosen so nothing needs clamping.
    ``STYLE_WB_CAP`` (styles.py) holds a style at the shipped x1 strength and
    ``WB_HARD_CAP`` is the renderer authority binding at EVERY strength, so
    the travel between them is what a style at its rail can still take."""
    assert WB_OPERATOR_SPAN == pytest.approx(WB_HARD_CAP - STYLE_WB_CAP)

    plan = dev.plan_variants(_write_photo(tmp_path / "IMG_0303.jpg"), list(STYLES))
    for name, params in plan.styled:
        for axis in ("white_balance_r", "white_balance_b"):
            reach = abs(getattr(params, axis)) + WB_OPERATOR_SPAN
            assert reach <= WB_HARD_CAP + 1e-9, f"{name}.{axis} + full travel outruns the renderer"


def test_a_neutral_run_moves_neither_a_styled_record_nor_the_video_path(tmp_path):
    """The bit-identity half. At the 0.0 defaults the operator block is skipped
    outright, so each variant is the ``apply_style`` record itself and the
    applier's neutrality guard still short-circuits; the video path reads no
    ``DevelopOptions``, so its cube still hashes to the pre-change digest."""
    plan = dev.plan_variants(_write_photo(tmp_path / "IMG_0304.jpg"), list(STYLES))
    assert plan.styled == [(n, apply_style(n, plan.base, plan.scene, 1.0)) for n in STYLES]

    ramp = np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], 3, axis=1)
    assert np.array_equal(cc._apply_colorbalance(ramp, cc.CorrectionParams()), ramp)

    cube = cc.params_to_lut3d_cube(VIDEO_PARAMS, cube_size=17, output_path=tmp_path / "v.cube")
    assert hashlib.sha256(cube.read_bytes()).hexdigest() == VIDEO_CUBE_SHA256
    assert "gs=0:" in build_filter_chain([VIDEO_PARAMS])


def test_a_look_can_name_every_axis_and_its_value_rides_the_measured_cast():
    """The look door, through the field names the two records SHARE — additive
    there too, because a look panel grades over the run's MEASURED base."""
    resolved = dev.look_variant_options(NEW_AXES, dev.DevelopOptions())
    laid = dev._look_params(
        cc.CorrectionParams(white_balance_r=0.0094, white_balance_b=-0.05), resolved
    )
    assert laid.white_balance_r == pytest.approx(0.0094 + NEW_AXES["white_balance_r"])
    assert laid.white_balance_b == pytest.approx(-0.05 + NEW_AXES["white_balance_b"])

    assert PhotosConfig(looks={"tungsten": {"white_balance_r": WB_OPERATOR_SPAN}}).looks
    with pytest.raises(ValidationError):
        PhotosConfig(looks={"tungsten": {"white_balance_r": WB_OPERATOR_SPAN + 0.01}})


def _run(argv: str) -> int:
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_every_axis_has_a_cli_door_and_out_of_range_is_refused_pre_decode(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(developed_output_dir=str(tmp_path / "out"))),
    )
    photo = _write_photo(tmp_path / "IMG_0305.jpg")
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    flags = " ".join(f"--{a.replace('_', '-')} {v}" for a, v in NEW_AXES.items())
    assert _run(f"photos develop {photo} --styles natural {flags}") == 0
    assert [getattr(seen[0], a) for a in NEW_AXES] == list(NEW_AXES.values())

    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad shift"))
    over = WB_OPERATOR_SPAN + 0.01
    assert _run(f"photos develop {photo} --white-balance-r {over}") == 1
    assert f"--white-balance-r must be in [{-WB_OPERATOR_SPAN:g}, {WB_OPERATOR_SPAN:g}]" in (
        capsys.readouterr().err
    )
