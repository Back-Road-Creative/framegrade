"""A photo whose ICC profile cannot be converted must not ship as a master.

The develop decode used to log "decoding untagged" and carry on, and the
exporters then tagged the never-converted pixels sRGB. Now the decode refuses
(``UnusableColorProfileError``) so no master or gallery file is written; a
profile-less source is unchanged (assumed sRGB) and a valid profile converts.
"""

import json
import pickle

import numpy as np
import pytest
from PIL import Image, ImageCms

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_module
from gomoveshift.pipeline.photos.develop import PhotoDecodeError, decode, srgb_icc_profile
from gomoveshift.pipeline.photos.resizer import PHOTO_PROVENANCE_NAME, PhotoResizer


def _write(path, icc=None):
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    kwargs = {"icc_profile": icc} if icc is not None else {}
    Image.fromarray(arr, "RGB").save(path, "JPEG", **kwargs)
    return path


def _valid_icc():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))


def _develop(photo, out):
    argv = f"photos develop {photo} --out {out}".split()
    return photos_handler.cmd_photos(_build_parser().parse_args(argv), None)


def test_decode_refuses_a_corrupt_profile(tmp_path):
    photo = _write(tmp_path / "bad.jpg", icc=b"not-a-real-icc-profile")
    with pytest.raises(develop_module.UnusableColorProfileError) as exc:
        decode(photo)
    err = exc.value
    assert isinstance(err, ValueError) and isinstance(err, PhotoDecodeError)
    assert err.path == photo
    assert str(photo) in str(err) and err.reason


def test_develop_writes_no_master_and_names_the_photo(tmp_path, capsys):
    photo = _write(tmp_path / "bad.jpg", icc=b"not-a-real-icc-profile")
    out = tmp_path / "out"
    assert _develop(photo, out) == 1
    assert not list(out.glob("*.jpg")) and not list(out.glob("*.tif"))
    err = capsys.readouterr().err
    assert f"! {photo}" in err and "colour profile" in err
    assert "1 of 1 photo(s) failed to develop" in err
    assert "could not decode this file" not in err


@pytest.mark.parametrize("icc", [None, "valid"])
def test_valid_and_profileless_sources_still_export_tagged_srgb(tmp_path, icc):
    photo = _write(tmp_path / "ok.jpg", icc=_valid_icc() if icc else None)
    out = tmp_path / "out"
    assert _develop(photo, out) == 0
    files = sorted(out.glob("*.jpg"))
    assert files
    for f in files:
        with Image.open(f) as im:
            assert im.info.get("icc_profile") == srgb_icc_profile()


def test_social_resize_records_the_assumed_srgb_note(tmp_path, caplog):
    bad = _write(tmp_path / "bad.jpg", icc=b"not-a-real-icc-profile")
    good = _write(tmp_path / "good.jpg", icc=_valid_icc())
    out = tmp_path / "social"
    resizer = PhotoResizer(output_base=out)
    with caplog.at_level("WARNING"):
        resizer.resize_batch([str(bad), str(good)], "2026-10-05")
    assert "assumed sRGB" in caplog.text
    record = json.loads((out / "2026-10-05" / PHOTO_PROVENANCE_NAME).read_text())
    by_name = {e["exported"]: e for e in record["photos"]}
    assert "assumed sRGB" in by_name["bad.jpg"]["color_assumption"]
    assert "color_assumption" not in by_name["good.jpg"]


def test_unusable_profile_error_survives_pickle(tmp_path):
    # The route-story still loader grades photos in a spawned process pool, which
    # ships a worker's error back by pickling it; the default reduce replays only
    # the message, so unpickling failed instead of naming the photo.
    err = develop_module.UnusableColorProfileError(tmp_path / "bad.jpg", "bad profile")
    clone = pickle.loads(pickle.dumps(err))  # noqa: S301 -- bytes this test just made
    assert type(clone) is develop_module.UnusableColorProfileError
    assert clone.path == err.path and clone.reason == err.reason
    assert str(clone) == str(err)
