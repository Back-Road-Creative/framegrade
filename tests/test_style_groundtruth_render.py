"""Corpus manifest + blind judging grid for the style ground-truth harness.

Pins the two properties the harness cannot be trusted without: the walk counts
CAPTURES (a stem, not a file — one capture ships as .JPG + .NEF + .dng), and the
grid is blind (panels captioned A/B/C, the mapping seeded off the capture id so
a re-run reproduces it). No real corpus, no GPU: sources are synthesised JPEGs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import style_groundtruth_render as r  # noqa: E402

STYLES = ("natural", "moody", "warm")
WIDE = ("natural", "vivid", "moody", "golden", "mono", "soft")


def _jpeg(path: Path, value: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 12, 3), value, np.uint8), "RGB").save(path, "JPEG")
    return path


def _variants(size=(60, 80)):
    return [(name, np.full((*size, 3), 20 * i, np.uint8)) for i, name in enumerate(STYLES)]


@pytest.fixture
def corpus(tmp_path):
    """Two captures, one of them a .JPG/.NEF/.dng triple, plus everything the
    walk must ignore: a top-level .cube and the gms_test/ + luts/ subdirs."""
    root = tmp_path / "captures"
    for stem in ("shot_0020", "shot_0021"):
        _jpeg(root / f"{stem}.JPG")
        (root / f"{stem}.NEF").write_bytes(b"not-a-real-raw")
    (root / "shot_0020.dng").write_bytes(b"not-a-real-raw")
    (root / "kodachrome.cube").write_text("TITLE\n")
    _jpeg(root / "gms_test" / "shot_0099.JPG")  # rendered pipeline output, not a source
    _jpeg(root / "luts" / "shot_0098.JPG")
    return root


def test_manifest_dedupes_by_stem_and_excludes_everything_that_is_not_a_source(corpus):
    manifest = r.build_manifest(corpus)
    assert manifest["schema"] == 1
    assert manifest["corpus_root"] == str(corpus.resolve())
    # The .JPG/.NEF/.dng triple counts ONCE; shot_0099 (gms_test/, pipeline output),
    # shot_0098 (luts/) and kodachrome.cube are not judgeable sources at all.
    assert [c["id"] for c in manifest["captures"]] == ["shot_0020", "shot_0021"]
    first = manifest["captures"][0]
    assert set(first) == {"id", "raw", "jpeg", "exif_dt"}
    assert first["raw"] == str((corpus / "shot_0020.NEF").resolve()), "prefer the raw"
    assert first["jpeg"] == str((corpus / "shot_0020.JPG").resolve())


def test_blind_panels_shuffle_is_seeded_by_capture_id():
    variants = _variants()
    panels = [(label, style) for label, style, _ in r.blind_panels("shot_0020", variants)]
    assert panels == [(label, style) for label, style, _ in r.blind_panels("shot_0020", variants)]
    assert [label for label, _ in panels] == ["A", "B", "C"]
    assert sorted(style for _, style in panels) == sorted(STYLES)
    ids = [f"shot_00{n}" for n in range(20, 26)]
    orders = {tuple(s for _, s, _ in r.blind_panels(i, variants)) for i in ids}
    assert len(orders) > 1, "the mapping must depend on the capture id, not be one fixed order"


def test_grid_holds_one_blind_panel_per_offered_variant(monkeypatch):
    grid = r.blind_grid(r.blind_panels("shot_0020", _variants()))
    assert grid.shape[1] == 3 * 80, "three panels side by side, each under the long-edge bound"
    assert grid.shape[0] > 60, "each row carries its own caption strip"

    from gomoveshift.pipeline.photos import develop

    seen: list[str] = []

    def _spy(variants, cols=None):
        seen.extend(name for name, _ in variants)
        return np.zeros((1, 1, 3), np.uint8)

    monkeypatch.setattr(develop, "compose_contact_sheet", _spy)
    r.blind_grid(r.blind_panels("shot_0020", _variants()))
    assert seen == ["A", "B", "C"], "a caption must never carry the style name"


def _capture(tmp_path: Path, cid: str = "shot_0020") -> dict:
    return {
        "id": cid,
        "raw": None,
        "jpeg": str(_jpeg(tmp_path / "cam" / f"{cid}.JPG")),
        "exif_dt": "2020-01-01T12:00:00",
    }


def _seams(monkeypatch, names=STYLES):
    """Spy the two seams equivalence is asserted on: the styles list handed to
    ``develop_variants`` and the shuffle seed handed to ``blind_panels``. Call
    capture, not pixel digests — JPEG bytes vary by libjpeg build."""
    from types import SimpleNamespace

    from gomoveshift.pipeline.photos import develop

    styles_seen: list = []
    seeds: list[str] = []

    def fake_develop_variants(source, styles=None):
        styles_seen.append(styles)
        frames = [(n, np.full((60, 80, 3), 20 * i, np.uint8)) for i, n in enumerate(names)]
        return [(n, SimpleNamespace(rgb=f)) for n, f in frames]

    real_blind_panels = r.blind_panels

    def spy_blind_panels(seed, variants):
        seeds.append(seed)
        return real_blind_panels(seed, variants)

    monkeypatch.setattr(develop, "develop_variants", fake_develop_variants)
    monkeypatch.setattr(r, "blind_panels", spy_blind_panels)
    return styles_seen, seeds


def test_render_capture_default_keeps_auto_selection_and_bare_seed(tmp_path, monkeypatch):
    styles_seen, seeds = _seams(monkeypatch)
    rec = r.render_capture(_capture(tmp_path), tmp_path / "out")
    assert styles_seen == [None], "no --styles: the shipped auto-selection stays under test"
    assert seeds == ["shot_0020"], "the default run keeps the bare capture-id seed"
    assert [p["label"] for p in rec["panels"]] == ["A", "B", "C"]


def test_render_capture_explicit_ladder_requests_it_and_prefixes_the_seed(tmp_path, monkeypatch):
    styles_seen, seeds = _seams(monkeypatch, names=WIDE)
    rec = r.render_capture(_capture(tmp_path), tmp_path / "out", styles=list(WIDE))
    assert styles_seen == [list(WIDE)], "the explicit ladder goes to develop_variants verbatim"
    assert seeds == ["wide-shot_0020"], "wide and narrow panel orderings must never collide"
    assert [p["label"] for p in rec["panels"]] == ["A", "B", "C", "D", "E", "F"]
    assert sorted(p["style"] for p in rec["panels"]) == sorted(WIDE)


def test_render_record_carries_scene_truth_and_exif_dt(tmp_path, monkeypatch):
    """The emitted profile EQUALS the one production scores — same construction
    as ``plan_variants``: the decoded frame plus the file's own EXIF."""
    from dataclasses import asdict, fields

    from gomoveshift.pipeline.photos.develop import decode
    from gomoveshift.pipeline.photos.exif_gps import extract_exif_capture_metadata
    from gomoveshift.pipeline.photos.scene import SceneProfile, compute_scene_profile

    _seams(monkeypatch)
    cap = _capture(tmp_path)
    rec = r.render_capture(cap, tmp_path / "out")
    assert rec["exif_dt"] == "2020-01-01T12:00:00"
    decoded = decode(cap["jpeg"])
    expected = compute_scene_profile(decoded.rgb, extract_exif_capture_metadata(decoded.path))
    assert rec["scene"] == asdict(expected)
    assert set(rec["scene"]) == {f.name for f in fields(SceneProfile)}, "the scorer's exact shape"


def test_render_cli_parses_the_ladder_and_defaults_to_auto(tmp_path, monkeypatch):
    import json

    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"schema": 1, "captures": [{"id": "X"}]}))
    calls: list = []

    def fake_render_capture(capture, out_dir, styles=None):
        calls.append((capture["id"], styles))
        return {"id": capture["id"]}

    monkeypatch.setattr(r, "render_capture", fake_render_capture)
    argv = ["render", "--manifest", str(mpath), "--out", str(tmp_path / "out")]
    assert r.main([*argv, "--styles", "natural, vivid,mono"]) == 0
    assert calls == [("X", ["natural", "vivid", "mono"])]
    calls.clear()
    assert r.main(argv) == 0
    assert calls == [("X", None)]
