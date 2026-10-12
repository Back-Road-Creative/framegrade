"""``photos.looks`` + ``photos develop --look NAME``: name a knob combination once.

``develop`` has ~20 operator knobs and no way to name a combination, so "moody at
1.4 with a teal split-tone and light grain" meant retyping eight flags exactly.
A ``photos.looks`` entry names one and ``--look`` recalls it. Precedence is
explicit flag > look > bare ``photos.<knob>`` default; an unconfigured registry
is inert (byte-identical); bad knobs fail at config LOAD naming the look. The
``.grade.json`` sidecar records what a frame WAS — a look declares what you WANT.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift
import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig

# One named look and the exact flags it stands in for — the equivalence the
# operator is promised, and what is checked by hand against a real NEF.
LOOK = {
    "styles": "moody",
    "style_strength": 1.4,
    "vignette": 0.35,
    "grain": 0.2,
    "split_tone_shadow": [0.0, 0.03, 0.08],
}
LOOK_AS_FLAGS = (
    "--styles moody --style-strength 1.4 --vignette 0.35 --grain 0.2 "
    "--split-tone-shadow 0.0 0.03 0.08"
)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _render(monkeypatch, photo, out, argv_extra="", **cfg):
    """Develop ``photo`` into ``out`` under ``PhotosConfig(**cfg)``; return every
    export as ``{filename: bytes}`` so a comparison covers the variant NAMES the
    look selected as well as the pixels."""
    _use_photos_config(monkeypatch, **cfg)
    assert _run(f"photos develop {photo} --out {out} {argv_extra}") == 0
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.jpg"))}


def test_unconfigured_registry_is_inert_and_byte_identical(tmp_path, monkeypatch):
    """The default is empty and no ``--look`` is passed: the shipped path, exactly."""
    assert PhotosConfig().looks == {}
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    bare = _render(monkeypatch, photo, tmp_path / "bare")
    configured = _render(monkeypatch, photo, tmp_path / "reg", looks={"filmish": LOOK})
    assert configured == bare, "a look nobody named must not touch a single byte"


def test_a_look_renders_exactly_as_its_knobs_passed_as_flags(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0002.jpg")
    by_flags = _render(monkeypatch, photo, tmp_path / "flags", LOOK_AS_FLAGS)
    by_look = _render(
        monkeypatch, photo, tmp_path / "look", "--look filmish", looks={"filmish": LOOK}
    )
    assert by_look == by_flags, "--look must be the flags, byte for byte"
    assert list(by_look) == ["IMG_0002_moody.jpg"], "the look selects the style too"
    assert by_look != _render(monkeypatch, photo, tmp_path / "plain", "--styles moody")


def test_precedence_flag_beats_look_beats_config_default(tmp_path, monkeypatch):
    """All three layers, each pinned to the render its winning value produces."""
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    deep = {"deep": {"vignette": 0.60}}
    at = {
        v: _render(monkeypatch, photo, tmp_path / f"at{v}", f"--styles natural --vignette {v}")
        for v in ("0.10", "0.30", "0.60")
    }

    cfg_only = _render(monkeypatch, photo, tmp_path / "cfg", "--styles natural", vignette=0.30)
    assert cfg_only == at["0.30"], "no look named: the bare config default still wins"
    look_wins = _render(
        monkeypatch,
        photo,
        tmp_path / "look",
        "--styles natural --look deep",
        vignette=0.30,
        looks=deep,
    )
    assert look_wins == at["0.60"], "the look beats the bare config default"
    flag_wins = _render(
        monkeypatch,
        photo,
        tmp_path / "flag",
        "--styles natural --look deep --vignette 0.10",
        vignette=0.30,
        looks=deep,
    )
    assert flag_wins == at["0.10"], "an explicitly-passed flag beats the look"


def test_unknown_knob_and_out_of_range_value_fail_at_config_load():
    """Loudly, at LOAD, naming the look and the offending key — never silently
    at render time, hours into a batch."""
    with pytest.raises(ValidationError) as unknown:
        PhotosConfig(looks={"moody14": {"vignetting": 0.3}})
    assert "moody14" in str(unknown.value) and "vignetting" in str(unknown.value)

    with pytest.raises(ValidationError) as blown:
        PhotosConfig(looks={"blown": {"vignette": 3.0}})
    assert "blown" in str(blown.value) and "vignette" in str(blown.value)


def test_unknown_look_is_refused_at_the_cli_door_listing_the_names(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0004.jpg")
    out = tmp_path / "d"
    _use_photos_config(monkeypatch, looks={"filmish": LOOK, "airy": {"clarity": 0.2}})
    assert _run(f"photos develop {photo} --out {out} --look bogus") == 1
    err = capsys.readouterr().err
    assert "bogus" in err and "filmish" in err and "airy" in err
    assert not list(out.glob("*.jpg")) if out.exists() else True, "nothing decoded or written"


def test_a_configured_look_survives_the_real_cli_entry_point(tmp_path):
    """Config validation must NOT import ``gomoveshift.pipeline.*``.

    ``Config.load`` runs at MODULE IMPORT time (``longform_metadata`` reads it to
    build two path constants), so a validator that imports the pipeline re-enters
    a half-initialised package: every command touching ``gomoveshift.pipeline``
    dies with an unrelated ``ImportError`` naming ``GOLDEN_COLUMNS`` — nothing
    that points the operator at their config file, and one look cost the whole
    CLI. pytest imports these modules in another order and hides it, so this runs
    a FRESH interpreter over a real ``config.yaml`` in the cwd, as a shell does.
    """
    photo = _write_photo(tmp_path / "IMG_0005.jpg")
    (tmp_path / "config.yaml").write_text(
        "photos:\n  looks:\n    filmish:\n      styles: moody\n      vignette: 0.35\n"
    )
    root = Path(gomoveshift.__file__).resolve().parent.parent
    entry = "from gomoveshift.cli import main; main()"
    argv = f"photos develop {photo} --look filmish --dry-run".split()
    proc = subprocess.run(
        [sys.executable, "-c", entry, *argv],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"a configured look broke the CLI:\n{proc.stderr}"
    assert "moody" in proc.stdout, proc.stdout


def _use_builtin_looks(monkeypatch, table):
    from gomoveshift.pipeline.photos import looks as builtin

    monkeypatch.setattr(builtin, "BUILTIN_LOOKS", table)


def test_every_builtin_look_names_real_knobs_with_in_range_values():
    """The gate that arms itself: an empty table iterates to a pass today and
    becomes the real check the moment a preset pack lands."""
    from gomoveshift.config import LOOK_KNOBS, LOOK_RANGES
    from gomoveshift.pipeline.photos.looks import BUILTIN_LOOKS

    for name, knobs in BUILTIN_LOOKS.items():
        for key, value in knobs.items():
            assert key in LOOK_KNOBS, f"built-in look '{name}': '{key}' is not a develop flag"
            if key in LOOK_RANGES:
                lo, hi = LOOK_RANGES[key]
                assert isinstance(value, (int, float)) and lo <= value <= hi, (
                    f"built-in look '{name}': {key}={value!r} must be in [{lo:g}, {hi:g}]"
                )


def test_the_builtin_table_passes_the_real_config_validator():
    """Not a re-implementation of the check above: this is the validator the
    operator's own config hits, so a preset can't ship that config would reject."""
    from gomoveshift.pipeline.photos.looks import BUILTIN_LOOKS

    assert PhotosConfig(looks=BUILTIN_LOOKS).looks == BUILTIN_LOOKS


def test_a_builtin_look_resolves_with_nothing_configured(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0006.jpg")
    _use_builtin_looks(monkeypatch, {"x": {"vignette": 0.2}})
    by_flag = _render(monkeypatch, photo, tmp_path / "flag", "--styles natural --vignette 0.2")
    by_look = _render(monkeypatch, photo, tmp_path / "look", "--styles natural --look x")
    assert by_look == by_flag, "a built-in look must render as its knobs passed as flags"


def test_a_configured_look_wins_a_name_collision_with_a_builtin(tmp_path, monkeypatch):
    """The merge is ``{**BUILTIN_LOOKS, **photos.looks}``: an operator's override
    has to survive an upgrade that ships a built-in of the same name."""
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    _use_builtin_looks(monkeypatch, {"x": {"vignette": 0.2}})
    theirs = _render(monkeypatch, photo, tmp_path / "theirs", "--styles natural --vignette 0.9")
    merged = _render(
        monkeypatch,
        photo,
        tmp_path / "merged",
        "--styles natural --look x",
        looks={"x": {"vignette": 0.9}},
    )
    assert merged == theirs, "the operator's config must win the name, not the built-in"


def test_the_unknown_look_refusal_tags_both_groups(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0008.jpg")
    out = tmp_path / "d"
    _use_builtin_looks(monkeypatch, {"cyberpunk": {"vignette": 0.2}})
    _use_photos_config(monkeypatch, looks={"airy": {"clarity": 0.2}})
    assert _run(f"photos develop {photo} --out {out} --look bogus") == 1
    err = capsys.readouterr().err
    assert "built-in: cyberpunk" in err and "configured: airy" in err, err

    _use_builtin_looks(monkeypatch, {})
    assert _run(f"photos develop {photo} --out {out} --look bogus") == 1
    assert "built-in: none" in capsys.readouterr().err, "an empty group reads 'none'"


def test_look_knobs_stay_in_sync_with_the_live_develop_knob_surface():
    """THE guard, and the reason ``LOOK_KNOBS`` may be a local literal: config
    validation must not import the pipeline (see the test above), so this is what
    proves the list stays correct. ``DevelopOptions`` has grown a field per PR;
    every field must be nameable in a look — mapped to a real ``photos develop``
    flag whose argparse default is ``None``, which is what the fill-only-if-unset
    precedence rides. A field with no ``LOOK_KNOBS`` entry and no alias fails
    here, not silently at render time."""
    # Test-time imports are outside the import cycle the validator has to avoid.
    from gomoveshift.config import LOOK_ALIASES, LOOK_KNOBS, LOOK_RUN_KNOBS
    from gomoveshift.pipeline.photos.develop import DevelopOptions

    # A RUN-owned knob (frame geometry) is named by its flag instead: a look is
    # refused it by name, and it still needs the None-default flag checked below.
    fields = [LOOK_ALIASES.get(f, f) for f in DevelopOptions.__dataclass_fields__]
    unreachable = [f for f in fields if f not in LOOK_KNOBS | LOOK_RUN_KNOBS]
    assert not unreachable, f"DevelopOptions fields no look or flag can name: {unreachable}"

    args = _build_parser().parse_args(["photos", "develop", "IMG_0001.jpg"])
    for knob in sorted(LOOK_KNOBS | (LOOK_RUN_KNOBS & set(fields))):
        assert hasattr(args, knob), f"look knob '{knob}' is not a `photos develop` flag"
        assert getattr(args, knob) is None, (
            f"look knob '{knob}' must default to None — a look fills only unset flags"
        )
