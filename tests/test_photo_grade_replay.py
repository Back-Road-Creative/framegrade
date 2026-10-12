"""``photos develop --from-grade``: an approved grade, replayed not re-derived.

The sidecar shipped write-only. This is the read half: the recorded record —
colour params AND the spatial knobs the frame was approved with — stands in for
the measure pass, so an approved proof re-renders byte for byte, or transfers to
the siblings shot in the same light. A foreign, malformed or differently-versioned
file is refused at the door, before any photo is decoded.
"""

import json

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
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))


def _write_photo(path, seed=11):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


#: A develop worth approving: one knob from each half of the record, so a replay
#: that restored only the colour params renders visibly different pixels.
KNOBS = "--vignette 0.4 --clarity 0.2 --grain 0.1 --sharpen 1.5 --vibrance 0.3"


def _approve(tmp_path, seed=11, name="IMG_0001"):
    """Develop one frame with a sidecar; return (photo, export, sidecar)."""
    photo = _write_photo(tmp_path / f"{name}.jpg", seed)
    out = tmp_path / f"{name}_proof"
    assert _run(f"photos develop {photo} --out {out} --styles natural --grade-sidecar {KNOBS}") == 0
    return photo, out / f"{name}_natural.jpg", out / f"{name}_natural.grade.json"


def test_replaying_an_approved_grade_reproduces_that_frame_byte_for_byte(tmp_path):
    """The deliverable: no knob is re-stated on the replay command line, and the
    export matches the approved one exactly — the recorded record IS the render."""
    photo, approved, sidecar = _approve(tmp_path)
    again = tmp_path / "again"
    assert _run(f"photos develop {photo} --out {again} --from-grade {sidecar}") == 0
    assert [p.name for p in sorted(again.iterdir())] == ["IMG_0001_natural.jpg"], (
        "a replay renders exactly the recorded variant, under its recorded name"
    )
    assert (again / "IMG_0001_natural.jpg").read_bytes() == approved.read_bytes()


def test_a_replay_at_a_new_format_still_carries_the_approved_numbers(tmp_path):
    """An approved JPEG proof, re-rendered as a 16-bit master: --format is no part
    of the record, so it applies to the replay rather than fighting it."""
    photo, _, sidecar = _approve(tmp_path)
    master = tmp_path / "master"
    argv = f"photos develop {photo} --out {master} --from-grade {sidecar} --format tiff16"
    assert _run(f"{argv} --grade-sidecar") == 0
    assert (master / "IMG_0001_natural.tif").is_file()
    replayed = json.loads((master / "IMG_0001_natural.grade.json").read_text())
    assert replayed["params"] == json.loads(sidecar.read_text())["params"]


def test_the_recorded_grade_transfers_to_a_sibling_frame(tmp_path):
    """The second real use: one photo's approved grade, applied to the others from
    the same light. The sibling is never metered — it renders the recorded numbers."""
    _, _, sidecar = _approve(tmp_path)
    sibling = _write_photo(tmp_path / "IMG_0002.jpg", seed=77)
    out = tmp_path / "sibling"
    assert _run(f"photos develop {sibling} --out {out} --from-grade {sidecar} --grade-sidecar") == 0
    approved, transferred = (
        json.loads(p.read_text()) for p in (sidecar, out / "IMG_0002_natural.grade.json")
    )
    assert transferred["params"] == approved["params"], "the grade transferred, not the metering"
    assert transferred["options"] == approved["options"], "the approved knobs came with it"
    assert transferred["source"] != approved["source"]


def test_an_explicit_knob_beats_the_file_it_replays(tmp_path):
    """Defined precedence, the one --look already establishes: a flag actually
    passed wins, everything else is restored from the record."""
    photo, approved, sidecar = _approve(tmp_path)
    out = tmp_path / "dialled"
    argv = f"photos develop {photo} --out {out} --from-grade {sidecar} --vignette 0.0"
    assert _run(f"{argv} --grade-sidecar") == 0
    assert (out / "IMG_0001_natural.jpg").read_bytes() != approved.read_bytes()
    recorded = json.loads((out / "IMG_0001_natural.grade.json").read_text())["options"]
    assert recorded["vignette"] == 0.0, "the flag won"
    assert recorded["grain"] == 0.1, "every knob it did not re-state came off the file"


def test_a_sidecar_from_another_algo_version_is_refused_before_any_decode(tmp_path, monkeypatch):
    """The provenance block exists for exactly this: another algo version cannot
    replay to the pixels the operator approved, so it is refused rather than
    quietly rendered — and refused upstream, before a photo is opened."""
    photo, _, sidecar = _approve(tmp_path)
    data = json.loads(sidecar.read_text())
    data["provenance"]["develop_algo_version"] = "0000deadbeef"
    sidecar.write_text(json.dumps(data))
    monkeypatch.setattr(dev, "decode", lambda *a, **k: pytest.fail("decoded a refused replay"))
    assert _run(f"photos develop {photo} --out {tmp_path / 'no'} --from-grade {sidecar}") == 1


@pytest.mark.parametrize(
    "text",
    ["{not json", '{"hello": 1}', '["a"]', '{"sidecar_version": "1"}', '{"provenance": "x"}'],
    ids=["unparseable", "foreign_object", "foreign_array", "missing_keys", "block_not_an_object"],
)
def test_a_foreign_or_malformed_file_is_refused_at_the_door(tmp_path, monkeypatch, text):
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    bad = tmp_path / "not.grade.json"
    bad.write_text(text)
    monkeypatch.setattr(dev, "decode", lambda *a, **k: pytest.fail("decoded a refused replay"))
    assert _run(f"photos develop {photo} --out {tmp_path / 'no'} --from-grade {bad}") == 1


def test_a_replay_refuses_to_share_its_run_with_another_surface(tmp_path, capsys):
    """A replay renders the ONE recorded variant. A flag asking for extra panels
    would silently render them off the measured base, so it is refused instead."""
    photo, _, sidecar = _approve(tmp_path)
    out = tmp_path / "mixed"
    argv = f"photos develop {photo} --out {out} --from-grade {sidecar} --look-variant polaroid"
    assert _run(argv) == 1
    assert "--from-grade" in capsys.readouterr().err
