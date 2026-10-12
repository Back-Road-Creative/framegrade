"""Score the shipped photo style ranker against judge labels.

A read-only instrument: it IMPORTS ``styles._score`` as the incumbent rather
than reimplementing it (a drifting copy would make the comparison meaningless)
and changes nothing under ``gomoveshift/``. Any ranker change it argues for is
a separate, later PR. It reads the harness manifest + labels JSON
(``schema: 1``) from disk rather than importing the sibling harness tools, so it
runs before they land: manifest captures carry ``scene`` (``SceneProfile``
fields), label rows carry ``offered``/``best``/``wrong``/``consensus``/``source``.
A ``consensus: "none"`` row and a capture with no scene profile leave EVERY
denominator and are counted in the report; a ``source: "human"`` row overrides the
``llm`` row for the same capture. ``--wide-labels`` takes a second labels file
from the 6-style wide round; a branch verdict requires it, because within one
file the judge only labels what it was offered.

Not covered here, and built on this file's helpers: the
trait-signal probe (how many disagreement captures a predicate on one of the
five scene traits that score nothing today would flip) and the leave-one-out
bake-off of candidate scorers with its >=2-capture winner margin. Both build on
``load_corpus``/``rank``/``measure`` here; neither is stubbed, because a
number nobody can check out-of-sample is worse than no number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from gomoveshift.pipeline.photos.develop import VARIANT_COUNT
from gomoveshift.pipeline.photos.scene import SceneProfile
from gomoveshift.pipeline.photos.styles import _FAVORED, STYLES, _score

SCHEMA_VERSION = 1
# Verdict bars as RATES so the counts scale with the corpus: wide-hit floor,
# wrong-offers ceiling per capture, alt-agreement floor.
RATES = (21 / 24, 1.0, 0.70)
_EPS = 1e-9
_ORDER = list(STYLES)
_SCENE_FIELDS = {f.name for f in fields(SceneProfile)}

Metrics = dict[str, tuple[int, int]]  # metric -> (k, n)


class Capture(NamedTuple):
    id: str
    offered: tuple[str, ...]
    best: str
    wrong: frozenset[str]
    scene: SceneProfile


class Corpus(NamedTuple):
    captures: tuple[Capture, ...] = ()
    no_consensus: int = 0
    no_scene: int = 0


def rank(scene: SceneProfile) -> list[str]:
    """The shipped selection shape: natural anchors, the rest rank by suitability
    with declaration order as the tie-break. Parity with ``select_styles`` is
    tested — this walks the FULL ranking, where ``select_styles`` truncates."""
    return ["natural", *sorted(_FAVORED, key=lambda n: (-_score(n, scene), _ORDER.index(n)))]


def _rows(labels: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in labels.get("captures", ()):
        prior = rows.get(row["id"], {})
        if prior.get("source") == "human" and row.get("source") != "human":
            continue  # a reviewed row is never overwritten by the LLM's
        rows[row["id"]] = row
    return rows


def load_corpus(manifest: Mapping[str, Any], labels: Mapping[str, Any]) -> Corpus:
    for doc, what in ((manifest, "manifest"), (labels, "labels")):
        if doc.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"{what}: unsupported schema {doc.get('schema')!r}")
    scenes = {
        e["id"]: SceneProfile(**{k: v for k, v in e["scene"].items() if k in _SCENE_FIELDS})
        for e in manifest.get("captures", ())
        if isinstance(e.get("scene"), Mapping)
    }
    rows = _rows(labels)
    kept: list[Capture] = []
    no_consensus = no_scene = 0
    for cid, row in rows.items():
        if row.get("consensus") == "none":
            no_consensus += 1
        elif cid not in scenes:
            no_scene += 1
        else:
            wrong = frozenset(row.get("wrong", ()))
            kept.append(
                Capture(cid, tuple(row.get("offered", ())), row["best"], wrong, scenes[cid])
            )
    return Corpus(tuple(kept), no_consensus, no_scene)


def measure_wide(labels: Mapping[str, Any], wide: Mapping[str, Any]) -> tuple[int, int]:
    """The real cross-file metric: over captures where BOTH the narrow and the
    6-style wide round reached consensus, did the wide round's free pick land
    inside what the ranker actually offered narrow? Within one file ``hit`` is
    vacuous — the judge can only pick from the offered panels."""
    if wide.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"wide labels: unsupported schema {wide.get('schema')!r}")
    narrow = _rows(labels)
    k = n = 0
    for cid, wrow in _rows(wide).items():
        nrow = narrow.get(cid)
        if nrow is None or "none" in (nrow.get("consensus"), wrow.get("consensus")):
            continue
        n += 1
        k += wrow["best"] in nrow.get("offered", ())
    return (k, n)


def measure(corpus: Corpus) -> Metrics:
    """Every rate over the consensus captures only. ``hit`` is an INTEGRITY
    check, not a quality signal: the judge only labels panels it was shown, so
    n/n is expected and a shortfall means a harness bug — it feeds no verdict
    branch (``measure_wide`` carries the real question). ``alt_rank_agreement``
    asks a narrower question than the others — among the HITS the judge did not
    answer with ``natural``, is the ranker's top alternative the one the judge
    chose? — so it carries its own, smaller denominator."""
    hit = natural = wrong = alt_k = alt_n = 0
    for cap in corpus.captures:
        on_target = cap.best in cap.offered
        hit += on_target
        natural += cap.best == "natural"
        wrong += sum(1 for name in cap.offered if name in cap.wrong)
        if on_target and cap.best != "natural":
            alt_n += 1
            alt_k += rank(cap.scene)[1] == cap.best
    n = len(corpus.captures)
    counts = ((hit, n), (natural, n), (alt_k, alt_n), (wrong, n * VARIANT_COUNT))
    return dict(zip(("hit", "natural_best_rate", "alt_rank_agreement", "wrong_offers"), counts))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval — at n~24 a bare percentage is a misleading presentation."""
    if n <= 0:
        return (0.0, 1.0)
    p, denom = k / n, 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def verdict(met: Metrics, rates: tuple[float, float, float] = RATES) -> str:
    """KEEP / PRUNE / SET, count bars scaled to the actual n so a growing corpus
    never silently loosens them. A branch REQUIRES the cross-file ``wide_hit``
    (the within-file ``hit`` is vacuous); without it the verdict is INCOMPLETE."""
    if "wide_hit" not in met:
        return "INCOMPLETE (wide labels required for a branch verdict)"
    n = met["hit"][1]
    wide_k, wide_n = met["wide_hit"]
    alt_k, alt_n = met["alt_rank_agreement"]
    wrong, wrong_n = met["wrong_offers"]
    wide_min = math.ceil(rates[0] * wide_n - _EPS)
    wrong_max = math.floor(rates[1] * n + _EPS)
    if wide_k < wide_min:
        branch = "SET"
    elif wrong > wrong_max or alt_k < rates[2] * alt_n - _EPS:
        branch = "PRUNE"
    else:
        branch = "KEEP"
    return (
        f"{branch} wide_hit={wide_k}/{wide_n} (min {wide_min}) "
        f"wrong={wrong}/{wrong_n} (max {wrong_max}) alt={alt_k}/{alt_n}"
    )


def render_report(corpus: Corpus, met: Metrics, rates: tuple[float, float, float] = RATES) -> str:
    lines = [
        "# Photo style ground-truth score",
        "",
        f"Consensus captures: {met['hit'][1]} (excluded: {corpus.no_consensus} with consensus "
        f"`none`, {corpus.no_scene} with no scene profile)",
        "",
        "| metric | k/n | rate | 95% Wilson CI |",
        "| --- | --- | --- | --- |",
    ]
    for label, (k, tot) in met.items():
        low, high = wilson(k, tot)
        pct = k / tot * 100 if tot else 0.0
        lines.append(f"| {label} | {k}/{tot} | {pct:.1f}% | {low * 100:.1f}-{high * 100:.1f}% |")
    lines += ["", f"VERDICT: {verdict(met, rates)}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument(
        "--wide-labels",
        type=Path,
        help="6-style wide-round labels (same schema); a branch verdict requires them",
    )
    ap.add_argument("--report", type=Path, help="also write the markdown report here")
    ap.add_argument("--rates", type=float, nargs=3, default=list(RATES), metavar="RATE")
    args = ap.parse_args(argv)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.manifest, args.labels)]
    corpus = load_corpus(*docs)
    if not corpus.captures:
        print(
            f"ERROR: 0 scorable captures ({corpus.no_scene} labeled rows have no `scene` "
            f"object, {corpus.no_consensus} no consensus) — every capture in the manifest "
            f"({args.manifest}) must carry a `scene` field with the SceneProfile values.",
            file=sys.stderr,
        )
        return 2
    met = measure(corpus)
    if args.wide_labels:
        wide = json.loads(args.wide_labels.read_text(encoding="utf-8"))
        met["wide_hit"] = measure_wide(docs[1], wide)
    rates = (args.rates[0], args.rates[1], args.rates[2])
    report = render_report(corpus, met, rates)
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
