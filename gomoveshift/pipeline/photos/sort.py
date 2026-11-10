"""``photos sort``: dedupe, classify and develop a whole folder into one fixed home.

Stages: inventory -> camera facts -> dedupe against the ledger -> classify ->
develop each unique photo into ``<run>/<category>/NNN-<stem>.jpg`` -> score the
developed JPEGs -> renumber each category by quality -> report. Without a scoring
hook the order stays capture order; ``NNN`` is the quality rank (1 = best) once
shots are scored. Quality decides order, never inclusion.
Outputs are private working JPEGs, so faces are not blurred. Originals are only
ever opened read-only. A photo that fails to develop is reported in
``skipped.csv`` and the run goes on; nothing drops a photo silently.
"""

from __future__ import annotations

import csv
import html
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from ... import __version__
from ..orchestrator_lock import GpuLockBusyError, acquire_gpu_lock, release_gpu_lock
from . import quality_order, sort_groups
from .aesthetic import AestheticModelUnavailable
from .capture_meta import CaptureRecord, read_capture_records
from .dedupe import Ledger, dedupe, sha256_file
from .develop import (
    SUPPORTED_EXTENSIONS,
    compute_batch_corrections,
    decode,
    develop,
    export_jpeg,
)
from .exif_gps import extract_exif_capture_metadata
from .ingest import _dedupe_raw_pairs, discover
from .subject import UNKNOWN_SUBJECT, CaptureOptics, SubjectCategory, recognise_photos

#: Rough seconds to develop one photo on one worker; only feeds the dry-run estimate.
DEVELOP_S_PER_PHOTO_ESTIMATE = 20.0
#: Rough seconds to score one photo's aesthetic on CPU (2 threads); dry-run estimate only.
AESTHETIC_S_PER_PHOTO_ESTIMATE = 3.0
#: Ledger score name for the aesthetic value; bump the suffix if the model changes.
AESTHETIC_SCORE_NAME = "aesthetic:v1"
AESTHETIC_THREADS = 2
#: Shots listed per category in the terminal summary.
TOP_N = 5
FETCH_MODELS_COMMAND = "gomoveshift photos fetch-models"
#: Paths handed to ``compute_batch_corrections`` at once, to bound memory.
CORRECTION_CHUNK = 60
WORKERS_MIN, WORKERS_MAX = 1, 4
NICE_INCREMENT = 10
LEDGER_NAME = "ledger.sqlite"
SIDECAR_SUFFIXES = frozenset({".xmp", ".json", ".pp3", ".dop", ".aae"})

CATEGORY_FOLDERS = {
    SubjectCategory.ASTRO: "night-sky",
    SubjectCategory.LANDSCAPE: "landscape",
    SubjectCategory.CLOSE_UP: "close-up",
    SubjectCategory.WILDLIFE: "wildlife",
    SubjectCategory.PORTRAIT: "portrait",
    SubjectCategory.LONG_EXPOSURE: "long-exposure",
    SubjectCategory.EXPOSURE_BRACKET: "hdr-source",
    SubjectCategory.UNKNOWN: "other",
}
RANKED_COLUMNS = (
    "category",
    "rank",
    "source",
    "output",
    "confidence",
    "evidence",
    "capture_time",
    "quality_score",
    "aesthetic_score",
    "technical",
    "category_score",
    "group_id",
    "group_best_rank",
)


class SortRefusalError(Exception):
    """The run must not start; nothing was decoded or written."""


@dataclass(frozen=True)
class SortOptions:
    dry_run: bool = False
    index_only: bool = False
    reprocess: bool = False
    workers: int = 2
    no_stacks: bool = False
    gpu: bool = False


@dataclass(frozen=True)
class DevelopTask:
    source: Path
    out: Path
    params: object
    cluster_index: int
    style: Optional[str]


def develop_one(task: DevelopTask) -> None:
    """Grade one photo (optionally through a routed style) and export a JPEG."""
    decoded = decode(task.source)
    params = task.params
    if task.style:
        from .scene import compute_scene_profile
        from .styles import apply_style

        scene = compute_scene_profile(decoded.rgb, extract_exif_capture_metadata(decoded.path))
        params = apply_style(task.style, params, scene)
    export_jpeg(develop(decoded, params, task.cluster_index).rgb, task.out)


@dataclass(frozen=True)
class SortHooks:
    """Seams for the slow or external stages; defaults are the real ones."""

    read_records: Callable = read_capture_records
    recognise: Callable = recognise_photos
    corrections: Callable = compute_batch_corrections
    develop: Callable = develop_one
    executor_factory: Callable = ProcessPoolExecutor
    #: ``score(rgb, category, *, aesthetic, path, exposure_s, focal_mm) -> ShotScore``;
    #: ``None`` leaves the category in capture order (see ``scoring_hooks``).
    score: Optional[Callable] = None
    #: ``factory(use_gpu=bool) -> scorer`` with ``score_unit(rgb) -> 0..1``; ``None`` = no aesthetic.
    aesthetic_factory: Optional[Callable] = None
    gpu_lock: Callable = acquire_gpu_lock
    gpu_release: Callable = release_gpu_lock
    #: ``sort_groups.GroupOps``: scene-group tags and HDR / panorama / night-stack outputs;
    #: ``None`` skips grouping (see ``scoring_hooks``).
    groups: Optional[sort_groups.GroupOps] = None


@dataclass(frozen=True)
class ShotScore:
    """One developed shot's score; the two part maps stay apart (``astro`` names ``stars`` in both)."""

    technical: float
    category: float
    aesthetic: Optional[float]
    total: float
    technical_parts: Mapping[str, float]
    category_parts: Mapping[str, float]


def score_shot(
    rgb, category: str, *, aesthetic=None, path=None, exposure_s=None, focal_mm=None
) -> ShotScore:
    night = category == quality_order.CATEGORY_NIGHT_SKY
    tech, tparts = quality_order.technical_score(
        rgb, night=night, path=path, exposure_s=exposure_s, focal_mm=focal_mm
    )
    cat, cparts = quality_order.category_signals(category, rgb)
    aes = None if aesthetic is None else min(1.0, max(0.0, float(aesthetic)))
    return ShotScore(tech, cat, aes, quality_order.combine(tech, cat, aes), tparts, cparts)


def make_aesthetic_scorer(*, use_gpu: bool = False):
    """The real local scorer, built once per run in the parent process."""
    from .aesthetic import AestheticScorer

    return AestheticScorer(use_gpu=use_gpu, threads=AESTHETIC_THREADS)


def scoring_hooks(**over) -> SortHooks:
    """The real hooks plus real scoring; the CLI runs with these."""
    over.setdefault("groups", sort_groups.real_ops())
    return SortHooks(score=score_shot, aesthetic_factory=make_aesthetic_scorer, **over)


@dataclass
class SortResult:
    run_dir: Optional[Path]
    counts: dict = field(default_factory=dict)
    estimate_s: Optional[float] = None


def clamp_workers(workers: int) -> int:
    return max(WORKERS_MIN, min(WORKERS_MAX, int(workers)))


def lower_priority() -> None:
    try:
        os.nice(NICE_INCREMENT)
    except OSError:
        pass


def check_paths(folder: Path, root: Path) -> None:
    if not folder.is_dir():
        raise SortRefusalError(f"{folder} does not exist or is not a folder")
    if not root.parent.is_dir():
        raise SortRefusalError(f"{root.parent} does not exist (is the output drive mounted?)")
    inner, outer = folder.resolve(), root.resolve()
    if outer.is_relative_to(inner) or inner.is_relative_to(outer):
        raise SortRefusalError(f"{folder} and the sort root {root} must not contain one another")


def _inventory(folder: Path):
    photos = discover(folder)
    kept = set(_dedupe_raw_pairs(photos)[0])
    skipped = [
        (p, "RAW+JPEG pair: developed from the RAW sibling") for p in photos if p not in kept
    ]
    photos = [p for p in photos if p in kept]
    for path in sorted(folder.rglob("*"), key=lambda p: str(p).lower()):
        suffix = path.suffix.lower()
        if path.is_file() and suffix not in SUPPORTED_EXTENSIONS and suffix not in SIDECAR_SUFFIXES:
            skipped.append((path, f"not a supported photo ({suffix or 'no extension'})"))
    return photos, skipped


def _unique_run_dir(root: Path, today: str, name: str) -> Path:
    base = root / "runs" / f"{today}_{name}"
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.name}_{n}")
    return candidate


def _capture_key(rec: CaptureRecord):
    return (rec.datetime_original or "", rec.subsec or "", str(rec.path))


def _optics(rec: CaptureRecord) -> CaptureOptics:
    return CaptureOptics(
        focal_length_mm=rec.focal_length_mm,
        f_number=rec.f_number,
        focus_distance_m=rec.focus_distance_m,
        lens_model=rec.lens_model,
    )


def _routed_style(reading) -> Optional[str]:
    from .styles import SUBJECT_ROUTING_CONFIDENCE_MIN, SUBJECT_STYLE_ROUTING

    ok = reading.confidence >= SUBJECT_ROUTING_CONFIDENCE_MIN
    return SUBJECT_STYLE_ROUTING.get(reading.category) if ok else None


def _write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])


def _thumb(rel: str, label: str, note: str = "") -> str:
    href = html.escape(quote(rel))
    return (
        f'<a href="{href}"><img src="{href}" alt="{html.escape(label)}" loading="lazy" '
        f'width="240"><span>{html.escape(label)}</span>{note}</a>'
    )


def _index_html(run_dir: Path, source: Path, ranked, outputs=()) -> str:
    best = sort_groups.best_by_group(ranked)
    sections = []
    folders = list(dict.fromkeys(row["category"] for row in ranked))
    if any(o.kind == "stacks" for o in outputs) and sort_groups.NIGHT_FOLDER not in folders:
        folders.append(sort_groups.NIGHT_FOLDER)
    for folder in folders:
        items = []
        if folder == sort_groups.NIGHT_FOLDER:
            items += [_thumb(o.rel, o.group_id) for o in outputs if o.kind == "stacks"]
        for row in (r for r in ranked if r["category"] == folder):
            name = Path(row["output"]).name
            total = row.get("quality_score")
            note = f"<small>{float(total):.2f}</small>" if total not in (None, "") else ""
            gid, top = row.get("group_id"), best.get(row.get("group_id"))
            if gid and top is not row and top is not None:
                note += f"<small>same scene as #{int(top['rank']):03d} ({html.escape(gid)})</small>"
            items.append(_thumb(f"{folder}/{name}", name, note))
        sections.append(
            f"<section><h2>{html.escape(folder)} ({len(items)})</h2>\n"
            + "\n".join(items)
            + "</section>"
        )
    for kind in ("hdr", "pano"):
        made = [o for o in outputs if o.kind == kind]
        if made:
            sections.append(
                f"<section><h2>{kind} ({len(made)})</h2>\n"
                + "\n".join(_thumb(o.rel, o.group_id) for o in made)
                + "</section>"
            )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(run_dir.name)}</title>"
        "</head><body>"
        f"<h1>{html.escape(run_dir.name)}</h1><p>{html.escape(str(source))}</p>\n"
        + "\n".join(sections)
        + "</body></html>\n"
    )


def _develop_all(plan, run_dir, hooks, workers, skipped):
    """Develop ``plan`` (``(path, folder, rank, reading)`` tuples); returns developed rows."""
    developed = []
    pending = []
    with hooks.executor_factory(workers) as pool:
        by_folder: dict[str, list] = {}
        for item in plan:
            by_folder.setdefault(item[1], []).append(item)
        for folder, items in by_folder.items():
            (run_dir / folder).mkdir(parents=True, exist_ok=True)
            for start in range(0, len(items), CORRECTION_CHUNK):
                chunk = items[start : start + CORRECTION_CHUNK]
                for item, params in zip(chunk, _corrections(chunk, hooks, skipped)):
                    if params is None:
                        continue
                    path, _, rank, reading = item
                    out = run_dir / folder / f"{rank:03d}-{path.stem}.jpg"
                    task = DevelopTask(path, out, params[0], params[1], _routed_style(reading))
                    pending.append((item, out, pool.submit(hooks.develop, task)))
        for item, out, future in pending:
            try:
                future.result()
            except Exception as exc:  # a file-format boundary: report, keep going
                skipped.append((item[0], f"develop failed: {exc}"))
                continue
            developed.append((item, out))
    return developed


def _read_rgb(path: Path):
    """The developed JPEG as uint8 RGB; read in the parent so the workers' memory stays bounded."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _aesthetic_value(scorer, ledger, sha_source: Path, rgb, stats) -> Optional[float]:
    """Cached by source sha256; a miss runs the model and stores the value."""
    try:
        sha = sha256_file(sha_source)
        hit = ledger.get_score(sha, AESTHETIC_SCORE_NAME)
        if hit is not None:
            stats["aesthetic_cached"] += 1
            return hit
        value = float(scorer.score_unit(rgb))
        ledger.put_score(sha, AESTHETIC_SCORE_NAME, value)
        stats["aesthetic_scored"] += 1
        return value
    except Exception:  # one image never drops the shot
        stats["aesthetic_failed"] += 1
        return None


def _open_aesthetic(hooks, options, warn):
    """``(scorer or None, lock or None)``; the GPU lock is taken only for ``--gpu``."""
    lock, use_gpu = None, False
    if options.gpu:
        try:
            lock = hooks.gpu_lock(wait_s=0)
            use_gpu = True
        except (GpuLockBusyError, OSError) as exc:
            warn(f"--gpu: the GPU lock is busy ({exc}); aesthetic scoring runs on CPU")
    try:
        return hooks.aesthetic_factory(use_gpu=use_gpu), lock
    except AestheticModelUnavailable as exc:
        warn(
            f"aesthetic model unavailable ({exc}); ordering without an aesthetic score. "
            f"Fetch the weights with: {FETCH_MODELS_COMMAND}"
        )
    except Exception as exc:  # e.g. onnxruntime cannot load the file
        warn(f"aesthetic model failed to load ({exc}); ordering without an aesthetic score")
    return None, lock


def _score_all(done, records, ledger, hooks, options, warn, hashes=None):
    """Score each developed JPEG; returns ``({source: ShotScore | None}, stats)``."""
    stats = {
        "aesthetic_cached": 0,
        "aesthetic_scored": 0,
        "aesthetic_failed": 0,
        "score_failed": 0,
    }
    scores: dict[Path, Optional[ShotScore]] = {}
    scorer, lock = None, None
    try:
        if hooks.aesthetic_factory is not None:
            scorer, lock = _open_aesthetic(hooks, options, warn)
        for (path, _name, _rank, reading), out in done:
            rec = records.get(path) or CaptureRecord(path=path)
            try:
                rgb = _read_rgb(out)
            except Exception:
                stats["score_failed"] += 1
                scores[path] = None
                continue
            if hashes is not None:
                try:
                    hashes[str(path)] = int(hooks.groups.phash(rgb))
                except Exception:  # a look-alike hash is optional
                    pass
            aesthetic = _aesthetic_value(scorer, ledger, path, rgb, stats) if scorer else None
            try:
                scores[path] = hooks.score(
                    rgb,
                    reading.category.value,
                    aesthetic=aesthetic,
                    path=path,
                    exposure_s=rec.exposure_s,
                    focal_mm=rec.focal_length_mm,
                )
            except Exception:
                stats["score_failed"] += 1
                scores[path] = None
    finally:
        if lock is not None:
            hooks.gpu_release(*lock)
    for key, label in (
        ("aesthetic_failed", "aesthetic score failed"),
        ("score_failed", "quality score failed"),
    ):
        if stats[key]:
            warn(f"{label} for {stats[key]} shot(s); they follow the scored ones in capture order")
    return scores, stats


def _renumber(renames) -> None:
    """Rename ``(old, new)`` pairs in place through temp names so no pair can collide."""
    moves = [(old, new) for old, new in renames if old != new]
    temps = []
    for i, (old, new) in enumerate(moves):
        tmp = old.with_name(f".renumber-{i}-{old.name}")
        os.replace(old, tmp)
        temps.append((tmp, new))
    for tmp, new in temps:
        os.replace(tmp, new)


def _rank_by_quality(done, scores):
    """Rank each category (total desc, then capture order for unscored); rename the files.

    Returns ranked-row dicts ordered by ``(category, new rank)``.
    """
    by_folder: dict[str, list] = {}
    for entry in done:
        by_folder.setdefault(entry[0][1], []).append(entry)
    rows, renames = [], []
    for folder, entries in by_folder.items():
        entries.sort(key=lambda e: e[0][2])
        scored = {str(e[0][0]): scores[e[0][0]] for e in entries if scores.get(e[0][0])}
        order = quality_order.order_category(scored)
        by_key = {str(e[0][0]): e for e in entries}
        sequence = [by_key[k] for k in order] + [e for e in entries if str(e[0][0]) not in scored]
        for rank, ((path, name, _old, reading), out) in enumerate(sequence, 1):
            final = out.with_name(f"{rank:03d}-{path.stem}.jpg")
            renames.append((out, final))
            rows.append((name, rank, path, final, reading))
    _renumber(renames)
    return rows


def _score_columns(rows, scores):
    tech = sorted({k for r in rows for k in (scores.get(r[2]) or _NO_SCORE).technical_parts})
    cat = sorted({k for r in rows for k in (scores.get(r[2]) or _NO_SCORE).category_parts})
    return tech, cat


_NO_SCORE = ShotScore(0.0, 0.0, None, 0.0, {}, {})


def _top_lines(ranked, limit: int = TOP_N) -> list[str]:
    lines = []
    for folder in dict.fromkeys(r["category"] for r in ranked):
        top = [r for r in ranked if r["category"] == folder and r.get("quality_score", "") != ""]
        if top:
            lines.append(f"  top {folder}:")
            lines += [
                f"    {r['rank']:>3}. {Path(r['output']).name}  {float(r['quality_score']):.2f}"
                for r in top[:limit]
            ]
    return lines


def _corrections(chunk, hooks, skipped):
    paths = [item[0] for item in chunk]
    try:
        return list(hooks.corrections(paths))
    except Exception:  # one bad frame must not sink its chunk: retry per file
        out = []
        for path in paths:
            try:
                out.append(hooks.corrections([path])[0])
            except Exception as exc:
                skipped.append((path, f"measure failed: {exc}"))
                out.append(None)
        return out


def run_sort(
    folder,
    root,
    options: SortOptions,
    hooks: Optional[SortHooks] = None,
    *,
    today: Optional[str] = None,
    say=print,
    resume=None,
) -> SortResult:
    started = time.monotonic()
    folder, root = Path(folder), Path(root)
    check_paths(folder, root)
    hooks = hooks or SortHooks()
    workers = clamp_workers(options.workers)
    today = today or date.today().isoformat()
    if resume is not None:
        return _resume(resume, folder, root, options, hooks, workers, say)
    timings: dict[str, float] = {}

    def lap(name, since):
        timings[name] = round(time.monotonic() - since, 3)
        return time.monotonic()

    mark = time.monotonic()
    photos, skipped = _inventory(folder)
    mark = lap("inventory", mark)
    records, warnings = hooks.read_records(photos)
    mark = lap("camera_facts", mark)

    ledger_file = root / LEDGER_NAME
    ledger = None
    if not options.dry_run:
        root.mkdir(exist_ok=True)
    if not options.dry_run or ledger_file.is_file():
        ledger = Ledger(ledger_file)
    try:
        result = dedupe(photos, records, ledger=ledger, reprocess=options.reprocess)
        mark = lap("dedupe", mark)
        run_id = f"index:{folder.name}:{today}" if options.index_only else None
        run_dir = None if options.dry_run else _unique_run_dir(root, today, folder.name)
        if run_dir is not None:
            run_id = run_id or run_dir.name
            run_dir.mkdir(parents=True)

        readings = {}
        if not options.index_only:
            readings = hooks.recognise(
                list(result.unique),
                optics={p: _optics(records[p]) for p in result.unique if p in records},
            )
            mark = lap("classify", mark)
        plan_by_folder: dict[str, list] = {}
        for path in result.unique:
            reading = readings.get(path, UNKNOWN_SUBJECT)
            plan_by_folder.setdefault(CATEGORY_FOLDERS[reading.category], []).append(
                (path, reading)
            )
        plan = []
        for name, members in plan_by_folder.items():
            members.sort(key=lambda m: _capture_key(records.get(m[0]) or CaptureRecord(path=m[0])))
            plan.extend((p, name, i, r) for i, (p, r) in enumerate(members, 1))
        planned = {name: len(members) for name, members in plan_by_folder.items()}

        warnings = list(warnings)

        def warn(message: str) -> None:
            warnings.append(message)
            say(f"warning: {message}")

        meta = {
            "source": str(folder),
            "framegrade_version": __version__,
            "run_id": run_id,
            "args": {**asdict(options), "workers": workers},
            "warnings": warnings,
        }
        estimate = len(plan) * DEVELOP_S_PER_PHOTO_ESTIMATE / workers
        if hooks.aesthetic_factory is not None:
            estimate += len(plan) * AESTHETIC_S_PER_PHOTO_ESTIMATE

        if options.dry_run:
            counts = {**planned, "duplicates": len(result.duplicates), "skipped": len(skipped)}
            _summary(say, planned, counts, time.monotonic() - started)
            say(
                f"Estimate: {estimate / 60:.1f} min to develop {len(plan)} photos "
                f"({DEVELOP_S_PER_PHOTO_ESTIMATE:g} s each / {workers} workers"
                + (
                    f", plus {AESTHETIC_S_PER_PHOTO_ESTIMATE:g} s each aesthetic scoring"
                    if hooks.aesthetic_factory is not None
                    else ""
                )
                + "). Nothing written."
            )
            return SortResult(None, counts, estimate)

        if options.index_only:
            ledger.record(run_id, _ledger_rows(result, result.unique))
            meta.update(counts={}, indexed=len(result.unique), duplicates=len(result.duplicates))
            _save_meta(run_dir, meta, timings, started)
            counts = {"indexed": len(result.unique), "duplicates": len(result.duplicates)}
            say(f"Indexed {len(result.unique)} shots ({len(result.duplicates)} duplicates).")
            return SortResult(run_dir, counts)

        done = _develop_all(plan, run_dir, hooks, workers, skipped)
        mark = lap("develop", mark)
        scores, stats, hashes = {}, None, None
        if hooks.score is not None:
            hashes = {} if hooks.groups is not None else None
            scores, stats = _score_all(done, records, ledger, hooks, options, warn, hashes)
            mark = lap("score", mark)
        ordered = _rank_by_quality(done, scores)
        tech_cols, cat_cols = _score_columns(ordered, scores)
        ranked = []
        for name, rank, path, out, reading in ordered:
            rec = records.get(path) or CaptureRecord(path=path)
            sc = scores.get(path)
            row = {
                "category": name,
                "rank": rank,
                "source": str(path),
                "output": str(out),
                "confidence": f"{reading.confidence:g}",
                "evidence": "; ".join(reading.evidence),
                "capture_time": rec.datetime_original or "",
            }
            if sc is not None:
                row["quality_score"] = f"{sc.total:.4f}"
                row["technical"] = f"{sc.technical:.4f}"
                row["category_score"] = f"{sc.category:.4f}"
                if sc.aesthetic is not None:
                    row["aesthetic_score"] = f"{sc.aesthetic:.4f}"
                for k, v in sc.technical_parts.items():
                    row[f"technical.{k}"] = f"{v:.4f}"
                for k, v in sc.category_parts.items():
                    row[f"category.{k}"] = f"{v:.4f}"
            ranked.append(row)
        if hooks.groups is not None:
            sort_groups.tag_groups(ranked, records, hooks.groups, hashes or {})
        columns = (*RANKED_COLUMNS, *(f"technical.{k}" for k in tech_cols))
        columns += tuple(f"category.{k}" for k in cat_cols)
        _write_csv(
            run_dir / "ranked.csv",
            columns,
            [[row.get(col, "") for col in columns] for row in ranked],
        )
        _write_csv(
            run_dir / "duplicates.csv",
            ("duplicate", "kept", "tier", "key", "reason", "seen_run"),
            [
                [h.duplicate, h.kept or "", h.tier.value, h.key, h.reason, h.seen_run or ""]
                for h in result.duplicates
            ],
        )
        _write_csv(run_dir / "skipped.csv", ("path", "reason"), [[p, r] for p, r in skipped])
        (run_dir / "index.html").write_text(_index_html(run_dir, folder, ranked), encoding="utf-8")
        # The gallery is complete and on disk: record the shots now, so a stack that is
        # interrupted later is resumed with ``resume=`` and never re-developed.
        ledger.record(run_id, _ledger_rows(result, [d[0][0] for d in done]))
        if hooks.groups is not None:
            meta["groups"] = _group_stage(
                run_dir, folder, ranked, skipped, hooks, options, ledger, workers, say
            )
            mark = lap("groups", mark)
        categories = dict(Counter(row["category"] for row in ranked))
        meta.update(
            counts=categories,
            developed=len(ranked),
            duplicates=len(result.duplicates),
            skipped=len(skipped),
        )
        if stats is not None:
            meta["scoring"] = stats
        _save_meta(run_dir, meta, timings, started)
        counts = {**categories, "duplicates": len(result.duplicates), "skipped": len(skipped)}
        _summary(say, categories, counts, time.monotonic() - started)
        for line in _top_lines(ranked):
            say(line)
        return SortResult(run_dir, counts)
    finally:
        if ledger is not None:
            ledger.close()


def _group_stage(run_dir, source, ranked, skipped, hooks, options, ledger, workers, say) -> dict:
    """Write the HDR / panorama / night-stack outputs; the page is rewritten after each lands."""

    def publish(outputs) -> None:
        (run_dir / "index.html").write_text(
            _index_html(run_dir, source, ranked, outputs), encoding="utf-8"
        )

    stage = sort_groups.build_outputs(
        run_dir,
        ranked,
        hooks.groups,
        ledger,
        workers=workers,
        no_stacks=options.no_stacks,
        on_output=publish,
    )
    skipped.extend(stage.skipped)
    _write_csv(run_dir / "skipped.csv", ("path", "reason"), [[p, r] for p, r in skipped])
    publish(stage.outputs)
    say(
        f"groups: {stage.stats['tagged']} tagged, {stage.stats['built']} built, "
        f"{stage.stats['resumed']} already done, {stage.stats['failed']} failed"
    )
    return dict(stage.stats)


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resume(run_dir, folder, root, options, hooks, workers, say) -> SortResult:
    """Finish the group outputs of an earlier run; nothing is re-developed or re-scored."""
    run_dir = Path(run_dir)
    if hooks.groups is None:
        raise SortRefusalError("--resume needs the group outputs (scoring hooks)")
    if not (run_dir / "ranked.csv").is_file() or root.resolve() not in run_dir.resolve().parents:
        raise SortRefusalError(f"{run_dir} is not a run folder under {root}")
    ranked = _read_rows(run_dir / "ranked.csv")
    skipped = [
        (r["path"], r["reason"])
        for r in _read_rows(run_dir / "skipped.csv")
        if not r["path"].startswith("group:")
    ]
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with Ledger(root / LEDGER_NAME) as ledger:
        meta["groups"] = _group_stage(
            run_dir, folder, ranked, skipped, hooks, options, ledger, workers, say
        )
    meta["skipped"] = len(skipped)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    counts = {**meta.get("counts", {}), "duplicates": meta.get("duplicates", 0)}
    return SortResult(run_dir, {**counts, "skipped": len(skipped)})


def _ledger_rows(result, paths):
    return [(result.keys[p], str(p), _sha(result.keys[p])) for p in paths]


def _sha(key: str) -> Optional[str]:
    return key.removeprefix("sha256:") if key.startswith("sha256:") else None


def _save_meta(run_dir, meta, timings, started) -> None:
    meta["timings"] = {**timings, "total_s": round(time.monotonic() - started, 3)}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _summary(say, categories, counts, elapsed: float) -> None:
    for name, n in categories.items():
        say(f"  {name}: {n}")
    say(f"{counts['duplicates']} duplicates, {counts['skipped']} skipped, elapsed {elapsed:.1f} s")
