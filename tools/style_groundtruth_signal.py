"""Signal over the style ground-truth labels — read-only; every ranking is the
SHIPPED ``_score``/tie-break with only the probed delta applied. (1) TRAIT
SIGNAL: net flips + split Cramér's V per dormant trait and EXIF capture hour.
(2) MONO CANDIDATES: C_delete and every ``incumbent AND one extra conjunct``
leaning, scored by mono-wrong captures un-stuck from first-backfill rank.
The ship rule lives in docs/style-groundtruth.md; applying it is a separate step."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import statistics
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

# Flat, not ``tools.``-qualified: the score module is imported under one name
# only, so a type checker does not see it twice.
from style_groundtruth_score import load_corpus, measure, rank

from gomoveshift.pipeline.photos.scene import SceneProfile
from gomoveshift.pipeline.photos.styles import _FAVORED, _LEANING, SCORE_LEANING, STYLES, _score

if TYPE_CHECKING:
    from style_groundtruth_score import Capture, Corpus

_ORDER = list(STYLES)
Pred = Callable[[Any], bool]
ScenePred = Callable[[SceneProfile], bool]
# C_delete shipped mono without a leaning; a missing incumbent NEVER fires, so
# its removal delta is 0 and every "incumbent AND …" conjunction is a no-op.
_INCUMBENT_MONO: ScenePred = _LEANING.get("mono", lambda s: False)


def dormant_traits() -> list[str]:
    """Fields absent from all _FAVORED/_LEANING/_score source — derived, never hardcoded."""
    fns: tuple[Callable[..., Any], ...] = (*_FAVORED.values(), *_LEANING.values(), _score)
    src = "".join(inspect.getsource(f) for f in fns)
    return [f.name for f in fields(SceneProfile) if not re.search(rf"\b(s|scene)\.{f.name}\b", src)]


def _rerank(scene: SceneProfile, override: Mapping[str, float]) -> list[str]:
    """Shipped ranking + score overrides; mirrors ``score.rank`` (declaration tie-break)."""
    order = sorted(_FAVORED, key=lambda n: (-override.get(n, _score(n, scene)), _ORDER.index(n)))
    return ["natural", *order]


def _pred(kind: str, v: Any) -> Pred:
    if kind == "==":
        return lambda x: bool(x == v)
    return lambda x: float(x) >= v if kind == ">=" else float(x) < v


def splits(values: Mapping[str, Any]) -> list[tuple[str, Pred]]:
    """Equality per observed value (categorical); both sides of observed midpoints (numeric)."""
    uniq = sorted(set(values.values()), key=repr)
    if uniq and isinstance(uniq[0], (bool, str)):
        return [(f"== {v!r}", _pred("==", v)) for v in uniq]
    nums = sorted(float(v) for v in uniq)
    mids = [(a + b) / 2 for a, b in zip(nums, nums[1:])]
    return [(f"{k} {m:g}", _pred(k, m)) for m in mids for k in (">=", "<")]


def _trait_row(caps: Sequence[Capture], values: Mapping[str, Any]) -> dict[str, Any]:
    """Best (split, judge-preferred style) by net flips; guarded like a real leaning."""

    def top(c: Capture, style: str, fire: bool) -> str:
        if not fire or _FAVORED[style](c.scene):
            return rank(c.scene)[1]
        return _rerank(c.scene, {style: _score(style, c.scene) + SCORE_LEANING})[1]

    styles = sorted({c.best for c in caps if rank(c.scene)[1] != c.best} & set(_FAVORED))
    scored: list[tuple[Any, ...]] = []
    for (desc, pred), style in ((s, y) for s in splits(values) for y in styles):
        fire = [c.id in values and pred(values[c.id]) for c in caps]
        was = [rank(c.scene)[1] == c.best for c in caps]
        now = [top(c, style, f) == c.best for c, f in zip(caps, fire)]
        gained = sum(n and not w for n, w in zip(now, was))
        broken = sum(w and not n for n, w in zip(now, was))
        scored.append((broken - gained, desc, style, gained, broken, fire))
    if not scored:
        return {"predicate": None, "style": None, "net_flips": 0, "gained": 0, "broken": 0, "effect_size": 0.0}  # fmt: skip
    _, desc, style, gained, broken, fire = min(scored, key=lambda t: t[:3])
    effect = round(_cramers_v(list(fire), [c.best for c in caps]), 3)
    return {"predicate": desc, "style": style, "net_flips": gained - broken, "gained": gained, "broken": broken, "effect_size": effect}  # fmt: skip


def _cramers_v(xs: list[Any], ys: list[str]) -> float:
    """V of the winning split's firing pattern vs the judge best — descriptive only."""
    if (k := min(len(set(xs)), len(set(ys)))) < 2:
        return 0.0
    chi2 = 0.0
    for xv, yv in ((x, y) for x in set(xs) for y in set(ys)):
        exp = xs.count(xv) * ys.count(yv) / len(xs)
        chi2 += (sum(1 for x, y in zip(xs, ys) if x == xv and y == yv) - exp) ** 2 / exp
    return math.sqrt(chi2 / (len(xs) * (k - 1)))


def trait_signal(corpus: Corpus, hrs: Mapping[str, int | None]) -> list[dict[str, Any]]:
    """One row per dormant trait plus EXIF capture hour (no-EXIF rows never fire)."""
    caps = [c for c in corpus.captures if c.best != "natural"]
    traits: dict[str, dict[str, Any]] = {
        t: {c.id: getattr(c.scene, t) for c in corpus.captures} for t in dormant_traits()
    }
    traits["capture_hour"] = {c.id: h for c in corpus.captures if (h := hrs.get(c.id)) is not None}
    return [{"trait": t, **_trait_row(caps, v)} for t, v in traits.items()]


def _mono_score(scene: SceneProfile, leaning: ScenePred | None) -> float:
    """Incumbent leaning swapped for ``leaning`` (None = C_delete); mirrors
    _score's not-favored guard so ONLY the leaning delta moves."""
    score = _score("mono", scene)
    if not _FAVORED["mono"](scene):
        score -= SCORE_LEANING * _INCUMBENT_MONO(scene)
        score += SCORE_LEANING * (leaning is not None and leaning(scene))
    return score


def conjuncts(corpus: Corpus) -> list[tuple[str, ScenePred]]:
    """Observed categorical (field, value) pairs + both sides of numeric quartiles."""

    def attr(field: str, pred: Pred) -> ScenePred:
        return lambda s: (v := getattr(s, field)) is not None and pred(v)

    out: list[tuple[str, ScenePred]] = []
    for f in fields(SceneProfile):
        vals = [v for c in corpus.captures if (v := getattr(c.scene, f.name)) is not None]
        if vals and isinstance(vals[0], (bool, str)):
            uniq = sorted(set(vals), key=repr)
            out += [(f"{f.name} == {v!r}", attr(f.name, _pred("==", v))) for v in uniq]
        elif vals:
            qs = statistics.quantiles(vals, n=4) if len(vals) > 1 else [float(vals[0])]
            for q, k in ((q, k) for q in sorted(set(qs)) for k in (">=", "<")):
                out.append((f"{f.name} {k} {q:g}", attr(f.name, _pred(k, q))))
    return out


def mono_candidates(corpus: Corpus) -> list[dict[str, Any]]:
    """C_delete + every ``incumbent AND one extra conjunct``, sorted by mono-wrong
    captures un-stuck from first-backfill (rank 4: 'no longer' presumes 'was'), then
    simplicity."""

    def both(extra: ScenePred) -> ScenePred:
        return lambda s: _INCUMBENT_MONO(s) and extra(s)

    wrong = [c for c in corpus.captures if "mono" in c.offered and "mono" in c.wrong]
    stuck = [c for c in wrong if rank(c.scene).index("mono") == 3]
    alt = [c for c in corpus.captures if c.best != "natural" and c.best in c.offered]
    base_alt = measure(corpus)["alt_rank_agreement"][0]
    cands: list[tuple[str, ScenePred | None, int]] = [("C_delete", None, 0)]
    cands += [(f"incumbent AND {d}", both(p), 3) for d, p in conjuncts(corpus)]
    rows: list[dict[str, Any]] = []
    for name, leaning, njunct in cands:
        pos = {c.id: _rerank(c.scene, {"mono": _mono_score(c.scene, leaning)}) for c in alt + stuck}  # fmt: skip
        rows.append({"candidate": name, "conjuncts": njunct,
                     "unstuck": sum(pos[c.id].index("mono") != 3 for c in stuck),
                     "delta_alt": sum(pos[c.id][1] == c.best for c in alt) - base_alt})  # fmt: skip
    rows.sort(key=lambda r: (-r["unstuck"], r["conjuncts"], r["candidate"]))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--report", type=Path, help="write the markdown stub here")
    ap.add_argument("--json", type=Path, dest="json_out", help="write the machine-readable doc")
    args = ap.parse_args(argv)
    mdoc, ldoc = (json.loads(p.read_text(encoding="utf-8")) for p in (args.manifest, args.labels))
    dts = {e["id"]: e.get("exif_dt") for e in mdoc.get("captures", ())}
    hours = {i: datetime.fromisoformat(d).hour if d else None for i, d in dts.items()}
    doc: dict[str, Any] = {"schema": 1, "dormant_traits": dormant_traits(),
        "trait_signal": trait_signal(corpus := load_corpus(mdoc, ldoc), hours),
        "mono_candidates": mono_candidates(corpus)}  # fmt: skip
    if args.json_out:
        args.json_out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    # The report is a deliberate stub — the ``--json`` doc is the load-bearing artifact.
    top = "{candidate} unstuck={unstuck} delta_alt={delta_alt}".format(**doc["mono_candidates"][0])
    dormant = ", ".join(doc["dormant_traits"])
    report = f"# Style ground-truth signal\n\nDormant traits: {dormant}\n\nBEST: {top}"
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
