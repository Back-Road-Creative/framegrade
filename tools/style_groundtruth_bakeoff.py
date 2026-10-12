"""Leave-one-out bake-off of candidate style scorers — read-only; nothing under
``gomoveshift/`` changes. Each candidate answers the shipped alt-rank-agreement
question OUT OF SAMPLE: for every held-out consensus capture whose judge best is
offered and not ``natural``, train on the rest only; a hit is the candidate's
top alternative (rank 1, after the natural anchor) equalling the judge's best.
``baseline`` is the shipped ranking exactly (``rank``, imported — never
reimplemented); ``predicates`` adds ONE SCORE_LEANING-sized nudge per style, its
best single-split predicate (enum equality / numeric midpoint) chosen on the
training fold only; ``fitted`` re-ranks by per-style least squares over the
numeric scene fields (target = judge-best indicator), natural still anchored.
The leakage test is built to catch a fold that saw its own held-out capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, get_type_hints

import numpy as np

from gomoveshift.pipeline.photos.scene import SceneProfile
from gomoveshift.pipeline.photos.styles import _FAVORED, SCORE_LEANING, STYLES, _score

# Import the sibling under its flat module name so it is not loaded twice under two names.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_groundtruth_score import Capture, Corpus, load_corpus, rank  # noqa: E402

WINNER_RULE = (
    "Winner rule (pre-committed): highest LOO agreement count; tie -> fewer "
    "parameters (baseline 0, predicates 1/style, fitted = weight count); a "
    "challenger must beat baseline by >= 2 captures or the incumbent stands."
)
_ORDER = list(STYLES)
_ALT_STYLES = tuple(n for n in _ORDER if n in _FAVORED)
_HINTS = get_type_hints(SceneProfile)
NUMERIC_FIELDS = tuple(sorted(n for n, t in _HINTS.items() if t is float))
_ENUM_FIELDS = tuple(sorted(n for n, t in _HINTS.items() if t is str or t is bool))
_WIDTH = len(NUMERIC_FIELDS) + 1  # numeric fields plus an intercept

Predicate = tuple[str, str, Any]
Predictor = Callable[[SceneProfile], str]


def _top(scores: Mapping[str, float]) -> str:
    """The shipped tie-break shape: score desc, then declaration order."""
    return min(_ALT_STYLES, key=lambda n: (-scores[n], _ORDER.index(n)))


def fit_baseline(train: Sequence[Capture]) -> Predictor:
    return lambda scene: rank(scene)[1]  # the shipped ranking; training ignored


def _holds(pred: Predicate, scene: SceneProfile) -> bool:
    field, op, value = pred
    v = getattr(scene, field)
    if op in ("==", "!="):
        return bool(v == value) if op == "==" else bool(v != value)
    return bool(v >= value) if op == ">=" else bool(v < value)


def _splits(train: Sequence[Capture]) -> list[Predicate]:
    """Both sides of every enum equality and numeric midpoint the fold supports."""
    preds: list[Predicate] = []
    for f in _ENUM_FIELDS:
        for v in sorted({getattr(c.scene, f) for c in train}, key=repr):
            preds += [(f, "==", v), (f, "!=", v)]
    for f in NUMERIC_FIELDS:
        vals = sorted({float(getattr(c.scene, f)) for c in train})
        for lo, hi in zip(vals, vals[1:]):
            preds += [(f, ">=", (lo + hi) / 2.0), (f, "<", (lo + hi) / 2.0)]
    return preds


def fit_predicates(train: Sequence[Capture]) -> Predictor:
    """Baseline plus ONE SCORE_LEANING-sized nudge per style: its best fold-chosen
    predicate, ties broken by (field, op, value)."""
    splits = _splits(train)
    chosen: dict[str, Predicate] = {}
    for style in _ALT_STYLES:
        if splits:
            chosen[style] = min(
                splits,
                key=lambda p: (
                    -sum(_holds(p, c.scene) == (c.best == style) for c in train),
                    p[0],
                    p[1],
                    repr(p[2]),
                ),
            )

    def predict(scene: SceneProfile) -> str:
        return _top(
            {
                n: _score(n, scene)
                + (SCORE_LEANING if n in chosen and _holds(chosen[n], scene) else 0.0)
                for n in _ALT_STYLES
            }
        )

    return predict


def _features(scene: SceneProfile) -> list[float]:
    return [float(getattr(scene, f)) for f in NUMERIC_FIELDS] + [1.0]


def fit_fitted(train: Sequence[Capture]) -> Predictor:
    """Per-style lstsq over numeric fields + intercept, target = judge-best indicator."""
    x = np.array([_features(c.scene) for c in train], dtype=float).reshape(len(train), _WIDTH)
    weights = {
        style: np.linalg.lstsq(x, np.array([float(c.best == style) for c in train]), rcond=None)[0]
        for style in _ALT_STYLES
    }

    def predict(scene: SceneProfile) -> str:
        feats = np.array(_features(scene))
        return _top({n: float(feats @ weights[n]) for n in _ALT_STYLES})

    return predict


CANDIDATES = {"baseline": fit_baseline, "predicates": fit_predicates, "fitted": fit_fitted}
PARAMS = {"baseline": 0, "predicates": len(_ALT_STYLES), "fitted": len(_ALT_STYLES) * _WIDTH}


def loo_agreement(corpus: Corpus) -> tuple[dict[str, int], int]:
    """LOO: each eligible capture is judged by candidates trained on all the rest."""
    eligible = [c for c in corpus.captures if c.best in c.offered and c.best != "natural"]
    counts = dict.fromkeys(CANDIDATES, 0)
    for held in eligible:
        train = [c for c in corpus.captures if c.id != held.id]
        for name, fit in CANDIDATES.items():
            counts[name] += fit(train)(held.scene) == held.best
    return counts, len(eligible)


def decide(counts: Mapping[str, int]) -> tuple[str, bool]:
    """WINNER_RULE as code; returns (winner, whether the >=2 margin was met)."""
    best = min(sorted(counts), key=lambda n: (-counts[n], PARAMS[n]))
    margin_met = best != "baseline" and counts[best] - counts["baseline"] >= 2
    return (best if margin_met else "baseline"), margin_met


def render_report(counts: Mapping[str, int], eligible: int, total: int) -> str:
    winner, margin_met = decide(counts)
    rows = "\n".join(f"| {n} | {counts[n]}/{eligible} | {PARAMS[n]} |" for n in counts)
    return (
        "# Style scorer bake-off (leave-one-out)\n\n"
        f"Eligible captures (judge best offered, not `natural`): {eligible} of {total} consensus."
        "\n\n| candidate | LOO agreement | parameters |\n| --- | --- | --- |\n"
        f"{rows}\n\n{WINNER_RULE}\n\nWINNER: {winner} (margin_met={margin_met})"
    )


def to_json(counts: Mapping[str, int], eligible: int) -> dict[str, Any]:
    winner, margin_met = decide(counts)
    return {
        "schema": 1,
        "winner_rule": WINNER_RULE,
        "eligible": eligible,
        "candidates": {n: {"loo_agreement": counts[n], "params": PARAMS[n]} for n in counts},
        "winner": winner,
        "margin_met": margin_met,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--report", type=Path, help="also write the markdown report here")
    ap.add_argument("--json", type=Path, help="also write the machine-readable result here")
    args = ap.parse_args(argv)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.manifest, args.labels)]
    corpus = load_corpus(*docs)
    counts, eligible = loo_agreement(corpus)
    report = render_report(counts, eligible, len(corpus.captures))
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps(to_json(counts, eligible), indent=2) + "\n", encoding="utf-8"
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
