"""The sRGB ICC tag must be the same bytes on every call, process and machine.

littleCMS stamps the wall clock into the ICC header's ``dateTime`` field, so a
freshly built profile differed second to second and NO two runs of the photo
pipeline produced a byte-equal JPEG for identical pixels — silently defeating
any byte-level reproducibility check, and exactly the class of non-determinism
the reserved ``DEVELOP_ALGO_VERSION`` / ``develop_cache_key`` contract exists to
protect. Both writers embed the tag (``export_jpeg``'s gallery master and the
resizer's social copy), so the fix belongs in the profile builder.
"""

import io
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

import gomoveshift.pipeline.photos.develop as dev

# ICC header dateTime: bytes 24-36, six big-endian uint16 (Y, M, D, h, m, s).
DATETIME_SLICE = slice(24, 36)
ZEROED_DATETIME = (0, 0, 0, 0, 0, 0)


def _frame(width=64, height=48):
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_two_calls_past_a_second_boundary_return_the_same_bytes():
    """The repro: a wall-clock stamp made the profile differ at byte 35."""
    first = dev.srgb_icc_profile()
    time.sleep(1.2)
    second = dev.srgb_icc_profile()

    differing = [i for i in range(len(first)) if first[i] != second[i]]
    assert first == second, f"ICC profile differs across a second boundary at {differing}"


def test_a_separate_process_builds_the_identical_profile():
    """A per-process cache is not enough — the pipeline runs as separate CLIs."""
    service_dir = Path(dev.__file__).resolve().parents[3]
    fresh = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gomoveshift.pipeline.photos.develop import srgb_icc_profile; "
            "print(srgb_icc_profile().hex())",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(service_dir)},
        cwd=service_dir,
        check=True,
    )
    assert fresh.stdout.strip() == dev.srgb_icc_profile().hex()


def test_header_datetime_is_normalised():
    stamp = struct.unpack(">6H", dev.srgb_icc_profile()[DATETIME_SLICE])
    assert stamp == ZEROED_DATETIME


def test_only_the_datetime_field_changed_and_it_is_still_sRGB():
    """Colorimetry is untouched: same primaries, whitepoint and transfer curve."""
    built = dev.srgb_icc_profile()
    reference = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    assert len(built) == len(reference)
    assert built[: DATETIME_SLICE.start] == reference[: DATETIME_SLICE.start]
    assert built[DATETIME_SLICE.stop :] == reference[DATETIME_SLICE.stop :]

    profile = ImageCms.ImageCmsProfile(io.BytesIO(built))
    assert "sRGB" in ImageCms.getProfileDescription(profile)
    assert profile.profile.xcolor_space.strip() == "RGB"


def test_two_exports_of_one_frame_are_byte_equal(tmp_path):
    """The payoff: identical pixels now encode to identical files."""
    frame = _frame()
    first = dev.export_jpeg(frame, tmp_path / "first.jpg")
    time.sleep(1.2)
    second = dev.export_jpeg(frame, tmp_path / "second.jpg")

    assert first.read_bytes() == second.read_bytes()
    with Image.open(second) as img:
        assert img.info.get("icc_profile") == dev.srgb_icc_profile()
