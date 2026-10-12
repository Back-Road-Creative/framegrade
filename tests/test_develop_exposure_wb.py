"""Tests for the develop-time exposure EV, highlight/shadow recovery, and
white-balance Kelvin/tint knobs.

Every field defaults to a no-op so ``DevelopOptions()`` / ``CorrectionParams()``
keep grading exactly as before this module existed.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams, _apply_curves
from gomoveshift.pipeline.photos.develop import DevelopOptions, _grade, _wb_shifted


def test_develop_options_defaults_are_all_no_ops():
    o = DevelopOptions()
    assert o.exposure_ev == 0.0
    assert o.shadow_recovery == 0.0
    assert o.highlight_recovery == 0.0
    assert o.white_balance_kelvin == 0.0
    assert o.white_balance_tint == 0.0


def test_correction_params_defaults_are_no_ops():
    p = CorrectionParams()
    assert p.exposure_ev == 0.0
    assert p.shadow_recovery == 0.0
    assert p.highlight_recovery == 0.0


def _flat_frame(color, size=(6, 8)):
    h, w = size
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_default_grade_is_unchanged_by_new_fields():
    rgb = _flat_frame((120, 100, 80))
    baseline = _grade(rgb, CorrectionParams())
    same = _grade(
        rgb, CorrectionParams(exposure_ev=0.0, shadow_recovery=0.0, highlight_recovery=0.0)
    )
    assert np.array_equal(baseline, same)


def test_positive_exposure_ev_brightens_the_frame():
    rgb = _flat_frame((100, 100, 100))
    base = _grade(rgb, CorrectionParams())
    brighter = _grade(rgb, CorrectionParams(exposure_ev=1.0))
    assert brighter[0, 0, 0] > base[0, 0, 0]


def test_negative_exposure_ev_darkens_the_frame():
    rgb = _flat_frame((100, 100, 100))
    base = _grade(rgb, CorrectionParams())
    darker = _grade(rgb, CorrectionParams(exposure_ev=-1.0))
    assert darker[0, 0, 0] < base[0, 0, 0]


def test_shadow_recovery_adds_onto_measured_shadow_lift():
    rgb = np.zeros((1, 3), dtype=np.float64)
    plain = _apply_curves(rgb, CorrectionParams(shadow_lift=5.0))
    recovered = _apply_curves(rgb, CorrectionParams(shadow_lift=5.0, shadow_recovery=10.0))
    assert recovered[0, 0] > plain[0, 0]


def test_highlight_recovery_adds_onto_measured_highlight_pull():
    rgb = np.ones((1, 3), dtype=np.float64)
    plain = _apply_curves(rgb, CorrectionParams(highlight_pull=5.0))
    recovered = _apply_curves(rgb, CorrectionParams(highlight_pull=5.0, highlight_recovery=10.0))
    assert recovered[0, 0] < plain[0, 0]


def test_kelvin_warms_the_r_b_axis_additively_on_top_of_measured_cast():
    measured = CorrectionParams(white_balance_r=0.01, white_balance_b=-0.01)
    warmer = _wb_shifted(measured, DevelopOptions(white_balance_kelvin=2000.0))
    assert warmer.white_balance_r > measured.white_balance_r
    assert warmer.white_balance_b < measured.white_balance_b


def test_tint_shifts_the_green_magenta_axis_additively():
    measured = CorrectionParams(white_balance_g=0.02)
    tinted = _wb_shifted(measured, DevelopOptions(white_balance_tint=0.05))
    assert tinted.white_balance_g == pytest.approx(0.07)


def test_no_kelvin_tint_is_a_byte_identical_no_op():
    measured = CorrectionParams(white_balance_r=0.01, white_balance_b=-0.01, white_balance_g=0.02)
    shifted = _wb_shifted(measured, DevelopOptions())
    assert shifted == measured


# ── CLI door: every new knob is a flag, gated against its registry span ──────

_CLI_KNOBS = {
    "white_balance_kelvin": (-500.0, 2500.0, "[-2000, 2000]"),
    "white_balance_tint": (0.2, 0.9, "[-0.5, 0.5]"),
    "exposure_ev": (1.5, 4.0, "[-3, 3]"),
    "shadow_recovery": (6.0, 25.0, "[0, 20]"),
    "highlight_recovery": (8.0, -1.0, "[0, 20]"),
}


def _run_cli(argv: str) -> int:
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.cli.parser import _build_parser

    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


@pytest.mark.parametrize("name", sorted(_CLI_KNOBS))
def test_each_new_knob_threads_from_its_flag_and_is_refused_out_of_range_pre_decode(
    name, tmp_path, monkeypatch, capsys
):
    """The handler's white-balance loop walks every ``white_balance*`` registry row,
    so a knob with no flag crashed EVERY ``photos develop`` run (AttributeError on
    the Namespace). Each knob needs its flag, threaded into the options record, and
    a bad value refused before any decode."""
    from PIL import Image

    import gomoveshift.config as config_module
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.config import Config, PhotosConfig
    from gomoveshift.pipeline.photos import develop as dev

    good, bad, span = _CLI_KNOBS[name]
    flag = "--" + name.replace("_", "-")
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(developed_output_dir=str(tmp_path / "out"))),
    )
    photo = tmp_path / "IMG_0303.jpg"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(photo, "JPEG")
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev, "develop_variants", lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1]
    )
    assert _run_cli(f"photos develop {photo} --styles natural {flag} {good}") == 0
    assert [getattr(o, name) for o in seen] == [good]

    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail(f"decoded despite a bad {flag}"))
    assert _run_cli(f"photos develop {photo} {flag}={bad}") == 1
    assert f"{flag} must be in {span}" in capsys.readouterr().err
