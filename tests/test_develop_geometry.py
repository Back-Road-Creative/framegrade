"""Tests for the develop-time geometry recipe fields: crop, straighten, and
horizon-level (the same rotate primitive as straighten, just a differently
measured angle).

Every field defaults to a no-op so ``DevelopOptions()`` keeps every existing
develop/develop_variants call byte-identical to before this module existed.
"""

from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos.develop import (
    DecodedPhoto,
    DevelopOptions,
    _crop,
    _straighten,
    develop_variants,
)


def _decoded(h=20, w=30):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, : w // 2] = (200, 60, 60)  # left half distinguishable from the right
    return DecodedPhoto(
        rgb=rgb,
        path=Path("fixture.jpg"),
        decoder="test",
        width=w,
        height=h,
        bit_depth=8,
        is_raw=False,
    )


def test_develop_options_geometry_defaults_are_no_ops():
    o = DevelopOptions()
    assert o.straighten_angle == 0.0
    assert o.crop_rect == ()


def test_straighten_zero_is_a_byte_identical_no_op():
    rgb = _decoded().rgb
    assert np.array_equal(_straighten(rgb, 0.0), rgb)


def test_crop_empty_is_a_byte_identical_no_op():
    rgb = _decoded().rgb
    assert np.array_equal(_crop(rgb, ()), rgb)


def test_straighten_rotates_the_frame():
    rgb = _decoded()
    rotated = _straighten(rgb.rgb, 90.0)
    assert rotated.shape == rgb.rgb.shape
    assert not np.array_equal(rotated, rgb.rgb)


def test_crop_trims_to_the_requested_fraction():
    rgb = _decoded(h=20, w=30).rgb
    cropped = _crop(rgb, (0.0, 0.0, 0.5, 1.0))
    assert cropped.shape == (20, 15, 3)


def test_default_develop_variants_output_is_unchanged_by_geometry_fields():
    decoded = _decoded()
    names_a, rgbs_a = zip(*[(n, v.rgb) for n, v in develop_variants(decoded, styles=["natural"])])
    names_b, rgbs_b = zip(
        *[
            (n, v.rgb)
            for n, v in develop_variants(
                decoded,
                styles=["natural"],
                options=DevelopOptions(straighten_angle=0.0, crop_rect=()),
            )
        ]
    )
    assert names_a == names_b
    for a, b in zip(rgbs_a, rgbs_b):
        assert np.array_equal(a, b)


def test_crop_rect_shrinks_every_developed_variant():
    decoded = _decoded()
    variants = list(
        develop_variants(
            decoded, styles=["natural"], options=DevelopOptions(crop_rect=(0.0, 0.0, 0.5, 1.0))
        )
    )
    assert variants, "expected at least one styled variant"
    for _, developed in variants:
        assert developed.rgb.shape[1] == decoded.width // 2
        assert developed.rgb.shape[0] == decoded.height


def test_straighten_angle_reaches_every_developed_variant():
    decoded = _decoded()
    baseline = dict(develop_variants(decoded, styles=["natural"]))
    straightened = dict(
        develop_variants(decoded, styles=["natural"], options=DevelopOptions(straighten_angle=45.0))
    )
    for name, developed in straightened.items():
        assert developed.rgb.shape == baseline[name].rgb.shape
        assert not np.array_equal(developed.rgb, baseline[name].rgb)


def _run_cli(argv: str) -> int:
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.cli.parser import _build_parser

    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _cli_photo(tmp_path, monkeypatch):
    """A tiny JPEG and a config pointing output at ``tmp_path`` — the same fixture
    shape ``test_develop_exposure_wb`` drives the flag door with."""
    from PIL import Image

    import gomoveshift.config as config_module
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.config import Config, PhotosConfig

    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(developed_output_dir=str(tmp_path / "out"))),
    )
    photo = tmp_path / "IMG_0404.jpg"
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(photo, "JPEG")
    return photo


def test_straighten_and_crop_thread_from_their_flags(tmp_path, monkeypatch):
    """The recipe fields are reachable from ``photos develop``: without a flag the
    geometry stage exists only for a caller that builds ``DevelopOptions`` by hand."""
    from gomoveshift.pipeline.photos import develop as dev

    photo = _cli_photo(tmp_path, monkeypatch)
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev, "develop_variants", lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1]
    )
    argv = f"photos develop {photo} --styles natural --straighten 1.5 --crop 0 0.1 0.9 1"
    assert _run_cli(argv) == 0
    assert [(o.straighten_angle, o.crop_rect) for o in seen] == [(1.5, (0.0, 0.1, 0.9, 1.0))]


@pytest.mark.parametrize(
    "flags, message",
    [
        ("--straighten=46", "--straighten must be in [-45, 45]"),
        ("--crop 0 0 1.2 1", "--crop values must each be in [0, 1]"),
        ("--crop 0.6 0 0.4 1", "--crop needs left < right and top < bottom"),
    ],
)
def test_a_bad_geometry_flag_is_refused_before_any_decode(
    flags, message, tmp_path, monkeypatch, capsys
):
    from gomoveshift.pipeline.photos import develop as dev

    photo = _cli_photo(tmp_path, monkeypatch)
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail(f"decoded despite {flags}"))
    assert _run_cli(f"photos develop {photo} {flags}") == 1
    assert message in capsys.readouterr().err


def test_a_look_cannot_carry_geometry():
    """Geometry is the RUN's: a look that reshaped the frame would make two variants
    of one photo incomparable, so a look naming either is refused at config load."""
    from pydantic import ValidationError

    from gomoveshift.config import PhotosConfig

    for key, value in (("straighten", 2.0), ("crop", [0.0, 0.0, 0.5, 1.0])):
        with pytest.raises(ValidationError, match="RUN's choice"):
            PhotosConfig(looks={"tilt": {key: value}})
