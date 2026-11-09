"""``photos sort``: dedupe, classify and develop a whole folder into one fixed home.

Stages: inventory -> camera facts -> dedupe against the ledger -> classify ->
develop each unique photo into ``<run>/<category>/NNN-<stem>.jpg`` -> report.
Order inside a category is capture order (a quality order is a later stage).
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from ... import __version__
from .capture_meta import CaptureRecord, read_capture_records
from .dedupe import Ledger, dedupe
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


def _index_html(run_dir: Path, source: Path, ranked) -> str:
    sections = []
    folders = list(dict.fromkeys(row["category"] for row in ranked))
    for folder in folders:
        items = []
        for row in (r for r in ranked if r["category"] == folder):
            name = Path(row["output"]).name
            rel = html.escape(quote(f"{folder}/{name}"))
            label = html.escape(name)
            items.append(
                f'<a href="{rel}"><img src="{rel}" alt="{label}" loading="lazy" '
                f'width="240"><span>{label}</span></a>'
            )
        sections.append(
            f"<section><h2>{html.escape(folder)} ({len(items)})</h2>\n"
            + "\n".join(items)
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
) -> SortResult:
    started = time.monotonic()
    folder, root = Path(folder), Path(root)
    check_paths(folder, root)
    hooks = hooks or SortHooks()
    workers = clamp_workers(options.workers)
    today = today or date.today().isoformat()
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

        meta = {
            "source": str(folder),
            "framegrade_version": __version__,
            "run_id": run_id,
            "args": {**asdict(options), "workers": workers},
            "warnings": list(warnings),
        }
        estimate = len(plan) * DEVELOP_S_PER_PHOTO_ESTIMATE / workers

        if options.dry_run:
            counts = {**planned, "duplicates": len(result.duplicates), "skipped": len(skipped)}
            _summary(say, planned, counts, time.monotonic() - started)
            say(
                f"Estimate: {estimate / 60:.1f} min to develop {len(plan)} photos "
                f"({DEVELOP_S_PER_PHOTO_ESTIMATE:g} s each / {workers} workers). Nothing written."
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
        lap("develop", mark)
        done.sort(key=lambda d: (d[0][1], d[0][2]))
        ranked = []
        for (path, name, rank, reading), out in done:
            rec = records.get(path) or CaptureRecord(path=path)
            ranked.append(
                {
                    "category": name,
                    "rank": rank,
                    "source": str(path),
                    "output": str(out),
                    "confidence": f"{reading.confidence:g}",
                    "evidence": "; ".join(reading.evidence),
                    "capture_time": rec.datetime_original or "",
                }
            )
        _write_csv(
            run_dir / "ranked.csv",
            RANKED_COLUMNS,
            [[row.get(col, "") for col in RANKED_COLUMNS] for row in ranked],
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
        categories = dict(Counter(row["category"] for row in ranked))
        meta.update(
            counts=categories,
            developed=len(ranked),
            duplicates=len(result.duplicates),
            skipped=len(skipped),
        )
        _save_meta(run_dir, meta, timings, started)
        ledger.record(run_id, _ledger_rows(result, [d[0][0] for d in done]))
        counts = {**categories, "duplicates": len(result.duplicates), "skipped": len(skipped)}
        _summary(say, categories, counts, time.monotonic() - started)
        return SortResult(run_dir, counts)
    finally:
        if ledger is not None:
            ledger.close()


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
