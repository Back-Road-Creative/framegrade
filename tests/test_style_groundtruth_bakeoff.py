"""Hermetic bake-off tests: hand fixtures only — no corpus on disk, no judge."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.photos.scene import SceneProfile  # noqa: E402
from tools import style_groundtruth_bakeoff as b  # noqa: E402
from tools.style_groundtruth_score import Corpus, load_corpus, rank  # noqa: E402

DEFAULTS: dict[str, Any] = {"is_near_monochrome": False}
DEFAULTS.update(dict.fromkeys("exposure_class sat_profile dynamic_range".split(), "normal"))
DEFAULTS.update(warmth="neutral", haze="clear")


def _cap(cid: str, best: str, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"id": cid, "best": best, "consensus": "3of3", "source": "llm"}
    row.update({"offered": ["natural", "moody", "mono"], "wrong": [], **kw})
    return row


def _corpus(rows: list[dict[str, Any]], scenes: dict[str, dict[str, Any]]) -> Corpus:
    captures = [{"id": cid, "scene": {**DEFAULTS, **sc}} for cid, sc in scenes.items()]
    return load_corpus(
        {"schema": 1, "corpus_root": "/corpus", "captures": captures},
        {"schema": 1, "judge": "test", "passes": 3, "captures": rows},
    )


# On fog the shipped tie-break ranks moody above mono, so a judge who keeps
# calling mono is what a data-driven nudge can learn — and the baseline cannot.
FOGGY_ROWS = [_cap(c, "mono") for c in "abc"]
FOGGY_SCENES: dict[str, dict[str, Any]] = {c: {"haze": "foggy"} for c in "abc"}


def test_baseline_is_the_shipped_top_alternative_and_ignores_training() -> None:
    for kw in ({}, {"haze": "foggy"}, {"warmth": "warm"}, {"exposure_class": "bright"}):
        scene = SceneProfile(**{**DEFAULTS, **kw})
        assert b.fit_baseline([])(scene) == rank(scene)[1]


def test_split_candidates_cover_enum_equality_and_numeric_midpoints() -> None:
    rows = [_cap("a", "moody"), _cap("b", "mono")]
    corpus = _corpus(rows, {"a": {"warm_fraction": 0.0}, "b": {"warm_fraction": 0.5}})
    splits = b._splits(list(corpus.captures))
    assert ("warm_fraction", ">=", 0.25) in splits and ("warm_fraction", "<", 0.25) in splits
    assert ("haze", "==", "clear") in splits and ("haze", "!=", "clear") in splits


def test_loo_predicates_learn_the_foggy_mono_nudge_the_baseline_cannot() -> None:
    counts, eligible = b.loo_agreement(_corpus(FOGGY_ROWS, FOGGY_SCENES))
    assert eligible == 3
    assert counts["baseline"] == 0  # shipped tie-break puts moody first on fog
    assert counts["predicates"] == 3  # one enum-equality nudge lifts mono to the top


def test_only_offered_non_natural_bests_enter_the_denominator() -> None:
    rows = [*FOGGY_ROWS, _cap("d", "natural"), _cap("e", "soft")]  # soft was never offered
    scenes: dict[str, dict[str, Any]] = {**FOGGY_SCENES, "d": {}, "e": {}}
    assert b.loo_agreement(_corpus(rows, scenes))[1] == 3


def test_a_capture_never_trains_the_fold_that_judges_it() -> None:
    rows = [_cap("a", "moody"), _cap("b", "moody"), _cap("x", "mono")]
    scenes = {"a": {"warm_fraction": 0.0}, "b": {"warm_fraction": 0.1}, "x": {"warm_fraction": 0.9}}
    corpus = _corpus(rows, scenes)
    caps = {c.id: c for c in corpus.captures}
    rest = [c for c in corpus.captures if c.id != "x"]
    # The fixture has teeth: leaking x into its own training fold flips x's prediction.
    assert b.fit_fitted(list(corpus.captures))(caps["x"].scene) == "mono"
    assert b.fit_fitted(rest)(caps["x"].scene) == "moody"
    # a and b agree out of sample; a leaked x would make this 3
    assert b.loo_agreement(corpus)[0]["fitted"] == 2


def test_winner_rule_margin_and_parameter_tiebreak() -> None:
    assert b.decide({"baseline": 4, "predicates": 6, "fitted": 3}) == ("predicates", True)
    assert b.decide({"baseline": 5, "predicates": 6, "fitted": 6}) == ("baseline", False)
    assert b.decide({"baseline": 4, "predicates": 6, "fitted": 6}) == ("predicates", True)
    assert b.decide({"baseline": 7, "predicates": 2, "fitted": 2}) == ("baseline", False)


def test_main_writes_json_and_report_both_carrying_the_precommitted_rule(tmp_path: Path) -> None:
    captures = [{"id": c, "scene": {**DEFAULTS, "haze": "foggy"}} for c in "abc"]
    manifest = {"schema": 1, "corpus_root": "/c", "captures": captures}
    labels = {"schema": 1, "judge": "t", "passes": 3, "captures": FOGGY_ROWS}
    paths = {n: tmp_path / n for n in ("m.json", "l.json", "report.md", "out.json")}
    paths["m.json"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["l.json"].write_text(json.dumps(labels), encoding="utf-8")
    argv = ["--manifest", str(paths["m.json"]), "--labels", str(paths["l.json"])]
    argv += ["--report", str(paths["report.md"]), "--json", str(paths["out.json"])]
    assert b.main(argv) == 0
    out = json.loads(paths["out.json"].read_text(encoding="utf-8"))
    assert out["candidates"]["baseline"] == {"loo_agreement": 0, "params": 0}
    assert out["candidates"]["predicates"] == {"loo_agreement": 3, "params": 5}
    assert (out["winner"], out["margin_met"]) == ("predicates", True)
    assert out["winner_rule"] == b.WINNER_RULE
    report = paths["report.md"].read_text(encoding="utf-8")
    assert b.WINNER_RULE in report and "WINNER: predicates (margin_met=True)" in report
