"""Hermetic tests for ``tools/style_groundtruth_score.py``: synthetic scenes and
hand-written label dicts only — no corpus on disk, no LLM judge, no rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.photos.scene import SceneProfile  # noqa: E402
from gomoveshift.pipeline.photos.styles import select_styles  # noqa: E402
from tools import style_groundtruth_score as s  # noqa: E402

DEFAULTS: dict[str, Any] = {"is_near_monochrome": False}
DEFAULTS.update(dict.fromkeys("exposure_class sat_profile dynamic_range".split(), "normal"))
DEFAULTS.update(warmth="neutral", haze="clear")
# Every bar sits exactly ON its threshold: wide_hit at the floor, wrong at the
# per-capture ceiling (24 of 72 slots), alt at 0.70 — one step past any of them flips.
KEEP: s.Metrics = {"hit": (24, 24), "natural_best_rate": (10, 24), "wrong_offers": (24, 72)}
KEEP.update(alt_rank_agreement=(7, 10), wide_hit=(21, 24))


def _cap(cid: str, best: str, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"id": cid, "best": best, "consensus": "3of3", "source": "llm"}
    row.update({"offered": ["natural", "moody", "vivid"], "wrong": [], **kw})
    return row


def _corpus(rows: list[dict[str, Any]], scenes: dict[str, dict[str, Any]]) -> s.Corpus:
    captures = [{"id": cid, "scene": {**DEFAULTS, **sc}} for cid, sc in scenes.items()]
    return s.load_corpus(
        {"schema": 1, "corpus_root": "/corpus", "captures": captures},
        {"schema": 1, "judge": "test", "passes": 3, "captures": rows},
    )


def test_rank_is_the_shipped_ranker_not_a_drifting_copy() -> None:
    for kw in ({}, {"haze": "foggy"}, {"warmth": "warm"}, {"exposure_class": "bright"}):
        scene = SceneProfile(**{**DEFAULTS, **kw})
        assert s.rank(scene)[:3] == select_styles(scene, 3)


def test_excluded_rows_leave_every_denominator_and_are_counted() -> None:
    rows = [_cap("a", "natural"), _cap("b", "moody"), _cap("c", "vivid", consensus="none")]
    rows.append(_cap("d", "soft"))  # metered by the judge, but carries no scene profile
    corpus = _corpus(rows, {"a": {}, "b": {}, "c": {}})
    assert [c.id for c in corpus.captures] == ["a", "b"]
    assert (corpus.no_consensus, corpus.no_scene) == (1, 1)
    assert s.measure(corpus)["hit"] == (2, 2)


def test_a_human_row_overrides_the_llm_row_for_the_same_capture() -> None:
    rows = [_cap("a", "vivid"), _cap("a", "moody", source="human"), _cap("a", "golden")]
    corpus = _corpus(rows, {"a": {}})
    assert [(c.id, c.best) for c in corpus.captures] == [("a", "moody")]


def test_metric_denominators_are_the_ones_each_question_deserves() -> None:
    rows = [
        _cap("a", "natural"),  # hit, natural
        _cap("b", "vivid", wrong=["moody"]),  # hit, alt: vivid IS the top alternative
        _cap("c", "moody", wrong=["vivid"]),  # hit, alt: moody is not
        _cap("d", "soft"),  # miss — soft was never offered
    ]
    met = s.measure(_corpus(rows, {c: {} for c in "abcd"}))
    assert met["hit"] == (3, 4)
    assert met["natural_best_rate"] == (1, 4)
    assert met["alt_rank_agreement"] == (1, 2)  # denominator: hits the judge did not call natural
    assert met["wrong_offers"] == (2, 12)


def test_wilson_interval_endpoints_on_a_known_case() -> None:
    low, high = s.wilson(1, 1)
    assert (round(low, 4), round(high, 4)) == (0.2065, 1.0)
    assert s.wilson(0, 0) == (0.0, 1.0)


def test_every_verdict_branch_fires_and_the_count_bars_scale_with_n() -> None:
    assert s.verdict(KEEP).startswith("KEEP")
    assert s.verdict({**KEEP, "wrong_offers": (25, 72)}).startswith("PRUNE")
    assert s.verdict({**KEEP, "alt_rank_agreement": (6, 10)}).startswith("PRUNE")
    assert s.verdict({**KEEP, "wide_hit": (20, 24)}).startswith("SET")
    # SET outranks PRUNE: a ranker that misses the wide round is replaced, not pruned
    assert s.verdict({**KEEP, "wide_hit": (20, 24), "wrong_offers": (30, 72)}).startswith("SET")
    grown: s.Metrics = {"hit": (48, 48), "natural_best_rate": (20, 48), "wrong_offers": (40, 144)}
    grown.update(alt_rank_agreement=(14, 20), wide_hit=(41, 48))
    assert s.verdict(grown).startswith("SET")  # 48 wide pairs need 42 hits, not the 21 that 24 did
    assert s.verdict({**grown, "wide_hit": (42, 48)}).startswith("KEEP")


def test_without_wide_labels_the_verdict_is_incomplete_never_a_branch() -> None:
    met = {k: v for k, v in KEEP.items() if k != "wide_hit"}
    assert s.verdict(met) == "INCOMPLETE (wide labels required for a branch verdict)"


def test_the_verdict_line_prints_every_count_it_used() -> None:
    met: s.Metrics = {"hit": (24, 24), "natural_best_rate": (10, 24), "wrong_offers": (34, 72)}
    met.update(alt_rank_agreement=(9, 11), wide_hit=(19, 23))
    assert s.verdict(met) == "SET wide_hit=19/23 (min 21) wrong=34/72 (max 24) alt=9/11"


def test_wide_hit_pairs_by_id_and_needs_consensus_on_both_sides() -> None:
    wide_offer = {"offered": ["natural", "moody", "vivid", "soft", "golden", "muted"]}
    narrow = {"schema": 1, "judge": "t", "passes": 3, "captures": [
        _cap("a", "natural"),
        _cap("b", "moody"),
        _cap("c", "vivid", consensus="none"),  # no narrow consensus -> pair excluded
        _cap("d", "soft"),  # no wide row at all -> pair excluded
    ]}  # fmt: skip
    wide = {"schema": 1, "judge": "t", "passes": 3, "captures": [
        _cap("a", "golden", **wide_offer),  # overridden by the human row below
        _cap("a", "moody", source="human", **wide_offer),  # in narrow's offered -> hit
        _cap("b", "golden", **wide_offer),  # never offered narrow -> miss
        _cap("c", "natural", **wide_offer),
        _cap("e", "natural", **wide_offer),  # no narrow row at all -> pair excluded
        _cap("f", "vivid", consensus="none", **wide_offer),  # no wide consensus
    ]}  # fmt: skip
    assert s.measure_wide(narrow, wide) == (1, 2)


def test_wide_labels_with_a_wrong_schema_are_refused() -> None:
    good = {"schema": 1, "judge": "t", "passes": 3, "captures": []}
    with pytest.raises(ValueError, match="wide labels"):
        s.measure_wide(good, {**good, "schema": 2})


def test_main_with_wide_labels_reports_wide_hit_and_a_branch_verdict(
    tmp_path: Path, capsys: Any
) -> None:
    manifest = {"schema": 1, "corpus_root": "/c", "captures": [{"id": "a", "scene": DEFAULTS}]}
    labels = {"schema": 1, "judge": "t", "passes": 3, "captures": [_cap("a", "natural")]}
    wide = {"schema": 1, "judge": "t", "passes": 3, "captures": [_cap("a", "moody")]}
    paths = {n: tmp_path / f"{n}.json" for n in ("manifest", "labels", "wide")}
    for name, doc in (("manifest", manifest), ("labels", labels), ("wide", wide)):
        paths[name].write_text(json.dumps(doc), encoding="utf-8")
    argv = ["--manifest", str(paths["manifest"]), "--labels", str(paths["labels"])]
    assert s.main([*argv, "--wide-labels", str(paths["wide"])]) == 0
    out = capsys.readouterr().out
    assert "| wide_hit | 1/1 |" in out
    assert "VERDICT: KEEP wide_hit=1/1 (min 1)" in out.splitlines()[-1]


def test_an_empty_corpus_fails_loud_naming_the_scene_field(tmp_path: Path, capsys: Any) -> None:
    manifest = {"schema": 1, "corpus_root": "/c", "captures": [{"id": "a"}]}  # no scene
    labels = {"schema": 1, "judge": "t", "passes": 3, "captures": [_cap("a", "natural")]}
    paths = [tmp_path / n for n in ("manifest.json", "labels.json")]
    paths[0].write_text(json.dumps(manifest), encoding="utf-8")
    paths[1].write_text(json.dumps(labels), encoding="utf-8")
    assert s.main(["--manifest", str(paths[0]), "--labels", str(paths[1])]) != 0
    err = capsys.readouterr().err
    assert "scene" in err and str(paths[0]) in err


def test_the_report_carries_the_exclusions_and_ends_on_one_greppable_verdict() -> None:
    rows = [_cap("a", "natural"), _cap("b", "vivid", consensus="none")]
    corpus = _corpus(rows, {"a": {}, "b": {}})
    report = s.render_report(corpus, s.measure(corpus))
    assert "excluded: 1 with consensus `none`, 0 with no scene profile" in report
    assert "| hit | 1/1 | 100.0% | 20.7-100.0% |" in report
    assert report.splitlines()[-1].startswith("VERDICT: ")


def test_main_reads_both_documents_and_writes_the_report(tmp_path: Path, capsys: Any) -> None:
    manifest = {"schema": 1, "corpus_root": "/c", "captures": [{"id": "a", "scene": DEFAULTS}]}
    labels = {"schema": 1, "judge": "t", "passes": 3, "captures": [_cap("a", "natural")]}
    paths = [tmp_path / name for name in ("manifest.json", "labels.json", "report.md")]
    paths[0].write_text(json.dumps(manifest), encoding="utf-8")
    paths[1].write_text(json.dumps(labels), encoding="utf-8")
    argv = ["--manifest", str(paths[0]), "--labels", str(paths[1]), "--report", str(paths[2])]
    assert s.main(argv) == 0
    assert paths[2].read_text(encoding="utf-8").splitlines()[-1].startswith("VERDICT: ")
    assert "VERDICT: " in capsys.readouterr().out
