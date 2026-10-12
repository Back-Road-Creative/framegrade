"""``photos develop --grade-sidecar``: the exact grade, recorded per frame.

A developed JPEG carries no record of the numbers that made it, so a frame the
operator likes cannot be reproduced or transferred. The sidecar writes those
numbers beside the export. Opt-in — off, nothing is written and the export is
byte-identical; a failed sidecar never costs the photo; two develops of one
frame produce byte-identical JSON.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as dev


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _write_photo(path, size=(48, 64)):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _write_cube(path, size=9):
    """A gentle film-ish cube: lifted, warm-biased blacks."""
    a = np.linspace(0.0, 1.0, size)
    rows = [f"{r * 0.9 + 0.09:.4f} {g * 0.9:.4f} {b * 0.8:.4f}" for b in a for g in a for r in a]
    path.write_text("\n".join([f"LUT_3D_SIZE {size}", *rows]) + "\n")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_off_by_default_writes_nothing_and_keeps_the_export_byte_identical(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    plain, carded = tmp_path / "plain", tmp_path / "carded"
    assert _run(f"photos develop {photo} --out {plain} --styles natural") == 0
    assert _run(f"photos develop {photo} --out {carded} --styles natural --grade-sidecar") == 0
    assert list(plain.glob("*.grade.json")) == [], "the default path writes no sidecar"
    assert (carded / "IMG_0001_natural.grade.json").is_file(), "the flag writes one"
    assert (plain / "IMG_0001_natural.jpg").read_bytes() == (
        carded / "IMG_0001_natural.jpg"
    ).read_bytes(), "the sidecar never touches the exported pixels"


def test_config_default_is_off_and_the_flag_or_config_turns_it_on(tmp_path, monkeypatch):
    assert PhotosConfig().grade_sidecar is False
    photo = _write_photo(tmp_path / "IMG_0002.jpg")
    out = tmp_path / "cfg"
    _use_photos_config(monkeypatch, developed_output_dir=str(out), grade_sidecar=True)
    assert _run(f"photos develop {photo} --styles natural") == 0
    assert (out / "IMG_0002_natural.grade.json").is_file(), "photos.grade_sidecar enables it"


def test_sidecar_round_trips_the_params_knobs_and_provenance_of_the_real_render(
    tmp_path, monkeypatch
):
    """Every recorded value is pinned to the frame the same options rendered —
    a stale default anywhere shows up as a mismatch here."""
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    out = tmp_path / "d"
    options = dev.DevelopOptions(vignette=0.4, clarity=0.2, grain=0.1, sharpen=1.5)
    name, developed = next(iter(dev.develop_variants(photo, ["natural"], 1.0, None, options)))
    argv = (
        f"photos develop {photo} --out {out} --styles natural --grade-sidecar "
        "--vignette 0.4 --clarity 0.2 --grain 0.1 --sharpen 1.5"
    )
    assert _run(argv) == 0

    data = json.loads((out / f"IMG_0003_{name}.grade.json").read_text())
    assert data["variant"] == "natural"
    assert data["style_strength"] == 1.0
    assert dev.correction_params_from_sidecar(data["params"]) == developed.params
    for knob in ("vignette", "clarity", "grain", "sharpen"):
        assert data["options"][knob] == getattr(options, knob), f"{knob} recorded as applied"
    assert data["measured"]["sharpen_amount"] == developed.sharpen_amount
    assert data["measured"]["denoise_h"] == developed.denoise_h
    assert data["provenance"] == {
        "decoder": "pil",
        "develop_algo_version": dev.DEVELOP_ALGO_VERSION,
        "develop_cache_key": dev.develop_cache_key(
            photo, developed.params, developed.decoder, options=options, depth=8
        ),
    }, "the reserved pair is READ for provenance — no cache is consulted or written"


def test_two_renders_of_one_photo_stamp_different_cache_keys(tmp_path, monkeypatch):
    """The knobs live OUTSIDE ``params``, so a key blind to them stamped the
    vignetted frame and the clean one identically — and a cache wired on it
    would hand back the wrong pixels."""
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    plain, veiled = tmp_path / "plain", tmp_path / "veiled"
    base = f"photos develop {photo} --styles natural --grade-sidecar --out"
    assert _run(f"{base} {plain}") == 0
    assert _run(f"{base} {veiled} --vignette 0.5") == 0
    keys = {
        json.loads((d / "IMG_0007_natural.grade.json").read_text())["provenance"][
            "develop_cache_key"
        ]
        for d in (plain, veiled)
    }
    assert len(keys) == 2


# Two looks separated by ONE spatial knob and nothing else: their colour chains
# are identical, so their ``params`` are too. Anything that tells the pair apart
# has to come from the look's OWN options record.
GRAIN_LOOKS = {"soft_grain": {"grain": 0.2}, "heavy_grain": {"grain": 0.6}}


def test_look_panels_differing_only_in_grain_record_their_own_options_and_keys(
    tmp_path, monkeypatch
):
    """A look panel renders through the look's own record, so that record is what
    the sidecar has to carry — handing every panel the RUN's stamped two visibly
    different renders with one options block and one cache key."""
    _use_photos_config(monkeypatch, looks=GRAIN_LOOKS)
    photo = _write_photo(tmp_path / "IMG_0008.jpg")
    out = tmp_path / "d"
    argv = (
        f"photos develop {photo} --out {out} --styles natural --grade-sidecar "
        "--look-variant soft_grain --look-variant heavy_grain"
    )
    assert _run(argv) == 0

    soft, heavy = (
        json.loads((out / f"IMG_0008_look_{n}.grade.json").read_text())
        for n in ("soft_grain", "heavy_grain")
    )
    assert soft["params"] == heavy["params"], "the pair is separated by no colour knob"
    assert (soft["options"]["grain"], heavy["options"]["grain"]) == (0.2, 0.6)
    assert soft["provenance"]["develop_cache_key"] != heavy["provenance"]["develop_cache_key"], (
        "two different renders cannot share one cache key"
    )


def test_a_look_panel_records_the_looks_knobs_and_a_styled_one_still_records_the_runs(
    tmp_path, monkeypatch
):
    _use_photos_config(monkeypatch, looks=GRAIN_LOOKS)
    photo = _write_photo(tmp_path / "IMG_0009.jpg")
    out, bare = tmp_path / "d", tmp_path / "bare"
    base = f"photos develop {photo} --styles natural --grade-sidecar --grain 0.05 --out"
    assert _run(f"{base} {out} --look-variant heavy_grain") == 0
    assert _run(f"{base} {bare}") == 0

    panel = json.loads((out / "IMG_0009_look_heavy_grain.grade.json").read_text())["options"]
    styled = json.loads((out / "IMG_0009_natural.grade.json").read_text())["options"]
    assert (panel["grain"], styled["grain"]) == (0.6, 0.05), "the look's knob, then the run's"
    assert {k: v for k, v in panel.items() if k != "grain"} == {
        k: v for k, v in styled.items() if k != "grain"
    }, "a panel owns its own knobs and nothing else"
    assert (out / "IMG_0009_natural.grade.json").read_bytes() == (
        bare / "IMG_0009_natural.grade.json"
    ).read_bytes(), "a non-look variant's sidecar is untouched by a look riding along"


def test_lut_surface_records_the_cube_name_strength_and_over_style(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0004.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    out = tmp_path / "d"
    argv = (
        f"photos develop {photo} --out {out} --styles natural --grade-sidecar "
        f"--lut {cube} --lut-strength 0.5 --lut-over moody"
    )
    assert _run(argv) == 0

    look = json.loads((out / "IMG_0004_portra_over_moody.grade.json").read_text())
    assert look["lut"] == {"name": "portra", "lut_strength": 0.5, "lut_over": "moody"}
    styled = json.loads((out / "IMG_0004_natural.grade.json").read_text())
    assert "lut" not in styled, "a styled variant is not a LUT surface"


def test_two_develops_of_one_frame_produce_byte_identical_sidecars(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0005.jpg")
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        assert _run(f"photos develop {photo} --out {out} --styles natural --grade-sidecar") == 0
    assert (first / "IMG_0005_natural.grade.json").read_bytes() == (
        second / "IMG_0005_natural.grade.json"
    ).read_bytes(), "sorted keys and stable floats keep the pair reproducible"


def test_sidecar_is_written_through_a_temp_file_then_renamed(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0006.jpg")
    out = tmp_path / "d"
    renames = []
    real_replace = os.replace
    monkeypatch.setattr(
        os,
        "replace",
        lambda src, dst: (renames.append((str(src), str(dst))), real_replace(src, dst))[1],
    )
    assert _run(f"photos develop {photo} --out {out} --styles natural --grade-sidecar") == 0
    sidecar = out / "IMG_0006_natural.grade.json"
    src = next(s for s, d in renames if d == str(sidecar))
    assert src.endswith(".tmp") and os.path.dirname(src) == str(out), "temp file, same directory"
    assert list(out.glob(".*.tmp")) == [], "no partial file survives"


def test_an_unwritable_sidecar_never_costs_the_photo(tmp_path, monkeypatch):
    """The export is the product; the sidecar is metadata. A blocked write logs
    loudly, leaves no partial file, and raises nothing."""
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    out = tmp_path / "d"
    out.mkdir()
    (out / "IMG_0007_natural.grade.json").mkdir()  # a directory the rename cannot replace
    assert _run(f"photos develop {photo} --out {out} --styles natural --grade-sidecar") == 0
    assert (out / "IMG_0007_natural.jpg").is_file(), "the JPEG still stands"
    assert list(out.glob(".*.tmp")) == [], "the temp file is cleaned up"


# ── style_kind: an artistic export is identifiable in the sidecar itself ──


def _fake_developed(path):
    from gomoveshift.pipeline.color_correction import CorrectionParams

    return dev.DevelopedPhoto(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        path=path,
        params=CorrectionParams(),
        cluster_index=0,
        decoder="pil",
    )


def test_a_natural_correction_panel_is_recorded_natural(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0100.jpg")
    payload = dev.grade_sidecar_payload(
        _fake_developed(photo), "natural", 1.0, dev.DevelopOptions(), depth=8
    )
    assert payload["style_kind"] == "natural"


def test_an_artistic_style_panel_is_recorded_artistic(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0101.jpg")
    payload = dev.grade_sidecar_payload(
        _fake_developed(photo), "moody", 1.0, dev.DevelopOptions(), depth=8
    )
    assert payload["style_kind"] == "artistic"


def test_a_lut_or_look_panel_is_recorded_artistic_without_asking_is_artistic_style(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0102.jpg")
    for variant in ("film_auto", "film_over_moody", "look_neon"):
        payload = dev.grade_sidecar_payload(
            _fake_developed(photo), variant, 1.0, dev.DevelopOptions(), depth=8
        )
        assert payload["style_kind"] == "artistic"
