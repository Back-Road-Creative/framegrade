"""Hermetic tests for ``tools/style_groundtruth_signal.py`` — hand-authored
manifest/label dicts only: no corpus on disk, no judge, no rendering."""

from __future__ import annotations

import json
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "tools"))

import style_groundtruth_signal as sig  # noqa: E402
from gomoveshift.pipeline.photos.scene import SceneProfile  # noqa: E402
from gomoveshift.pipeline.photos.styles import _score  # noqa: E402
from style_groundtruth_score import Corpus, load_corpus, rank  # noqa: E402

DEFAULTS: dict[str, Any] = {"is_near_monochrome": False}
DEFAULTS.update(dict.fromkeys("exposure_class sat_profile dynamic_range".split(), "normal"))
DEFAULTS.update(warmth="neutral", haze="clear")
# Any OTHER still-required field is filled mechanically off the live dataclass:
# a sibling change deletes several SceneProfile fields, and keying the defaults
# this way keeps the fixtures green on both sides of that deletion.
DEFAULTS.update(
    {f.name: 0.0 for f in fields(SceneProfile) if f.name not in DEFAULTS and f.default is MISSING}
)
# bright+muted+flat: vivid 3, soft 3, moody 1, golden 1, mono 1 — the shape that used to
# stick mono 4th via the deleted muted+flat leaning; unfavored mono now ranks last.
BACKFILL = {"exposure_class": "bright", "sat_profile": "muted", "dynamic_range": "flat"}
TRIO, MONO3 = ["natural", "vivid", "moody"], ["natural", "vivid", "mono"]


def _row(cid: str, best: str, offered: list[str], wrong: list[str] | None = None) -> dict[str, Any]:
    row = {"id": cid, "best": best, "offered": offered, "consensus": "3of3", "source": "llm"}
    return {**row, "wrong": wrong or []}


def _corpus(rows: list[dict[str, Any]], scenes: dict[str, dict[str, Any]]) -> Corpus:
    captures = [{"id": cid, "scene": {**DEFAULTS, **sc}} for cid, sc in scenes.items()]
    return load_corpus({"schema": 1, "captures": captures}, {"schema": 1, "captures": rows})


def test_dormant_traits_are_derived_from_predicate_source_not_hardcoded() -> None:
    dormant = sig.dormant_traits()
    # dynamic_range went dormant when mono's muted+flat leaning was deleted
    assert "warm_fraction" in dormant and "dynamic_range" in dormant
    used = "exposure_class warmth sat_profile haze is_near_monochrome"
    assert not {*used.split(), "chroma_headroom", "capture_hour"} & set(dormant)


def test_rerank_with_no_override_is_the_shipped_ranking() -> None:
    for kw in ({}, BACKFILL, {"haze": "foggy"}):
        scene = SceneProfile(**{**DEFAULTS, **kw})
        assert sig._rerank(scene, {}) == rank(scene)


def test_trait_signal_reports_gains_breaks_and_split_effect() -> None:
    # a/b: judge best moody, shipped top-alt vivid (disagreements); c/d agree on vivid.
    # dynamic_range is dormant post-C_delete, so "flat" here never moves the shipped rank.
    rows = [_row(c, b, TRIO) for c, b in zip("abcd", ("moody", "moody", "vivid", "vivid"))]
    flat = {"dynamic_range": "flat"}
    corpus = _corpus(rows, {"a": flat, "b": flat, "c": flat, "d": {}})
    by = {r["trait"]: r for r in sig.trait_signal(corpus, {"a": 19, "b": 19, "c": 12, "d": None})}
    assert set(by) == {*sig.dormant_traits(), "capture_hour"}
    best = by["capture_hour"]  # a clean split: nudging moody flips a and b, breaks nothing;
    # hours come from exif_dt, so d's None row never fires
    assert (best["predicate"], best["style"], best["net_flips"]) == (">= 15.5", "moody", 2)
    assert (best["gained"], best["broken"], best["effect_size"]) == (2, 0, 1.0)
    dr = by["dynamic_range"]  # its winning split also fires on the agreeing c: one break
    assert (dr["net_flips"], dr["gained"], dr["broken"]) == (1, 2, 1)
    assert dr["predicate"] == "== 'flat'"
    assert by["warm_fraction"]["predicate"] is None  # constant trait: nothing to split on


def test_mono_candidates_model_the_missing_incumbent_as_never_firing() -> None:
    # C_delete SHIPPED: with no incumbent leaning left to remove, its delta is 0 —
    backfill = SceneProfile(**{**DEFAULTS, **BACKFILL})
    assert sig._mono_score(backfill, None) == _score("mono", backfill) == 1.0
    # — while a candidate leaning still adds its bump, and only when unfavored.
    assert sig._mono_score(backfill, lambda s: True) == 2.0
    foggy = SceneProfile(**{**DEFAULTS, **BACKFILL, "haze": "foggy"})  # favored: guard holds
    assert sig._mono_score(foggy, lambda s: True) == _score("mono", foggy) == 3.0
    rows = [_row("m1", "vivid", MONO3, ["mono"]), _row("m2", "vivid", MONO3, ["mono"]),
            _row("m3", "soft", ["natural", "mono", "soft"], ["mono"]), _row("n1", "vivid", TRIO)]  # fmt: skip
    muted_flat = {"sat_profile": "muted", "dynamic_range": "flat"}
    corpus = _corpus(rows, {"m1": BACKFILL, "m2": BACKFILL, "m3": muted_flat, "n1": {}})
    cands = sig.mono_candidates(corpus)
    # nothing sits 4th any more and every "incumbent AND …" conjunction can never
    # fire, so all rows are no-ops and the simplest candidate stays first
    assert (cands[0]["candidate"], cands[0]["conjuncts"]) == ("C_delete", 0)
    assert {(r["unstuck"], r["delta_alt"]) for r in cands} == {(0, 0)}
    same = {r["candidate"]: r for r in cands}["incumbent AND haze == 'clear'"]
    assert same["conjuncts"] == 3


def test_main_writes_the_json_doc_and_a_greppable_report(tmp_path: Path) -> None:
    caps = [{"id": "m1", "scene": {**DEFAULTS, **BACKFILL}, "exif_dt": "2020-01-01T12:00:00"}]
    labels = {"schema": 1, "captures": [_row("m1", "vivid", MONO3, ["mono"])]}
    paths = [tmp_path / n for n in ("m.json", "l.json", "report.md", "signal.json")]
    paths[0].write_text(json.dumps({"schema": 1, "captures": caps}), encoding="utf-8")
    paths[1].write_text(json.dumps(labels), encoding="utf-8")
    flags = ("--manifest", "--labels", "--report", "--json")
    assert sig.main([a for f, p in zip(flags, paths) for a in (f, str(p))]) == 0
    doc = json.loads(paths[3].read_text(encoding="utf-8"))
    assert (doc["schema"], doc["dormant_traits"]) == (1, sig.dormant_traits())
    assert len(doc["trait_signal"]) == len(sig.dormant_traits()) + 1  # + EXIF capture hour
    assert doc["mono_candidates"][0]["candidate"] == "C_delete"  # all no-ops: simplest first
    assert paths[2].read_text(encoding="utf-8").splitlines()[-1].startswith("BEST: ")
