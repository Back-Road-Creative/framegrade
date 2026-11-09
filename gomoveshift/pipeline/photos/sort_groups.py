"""Scene groups for ``photos sort``: tags on ``ranked.csv`` plus HDR, panorama and night-stack outputs.

A group is a tag: every member stays in its own category at its own rank. Kinds, in the
order a shot is claimed: ``night`` (an intervalometer chunk), ``bracket`` (exposure
bracket), ``pano`` (a fast run of same-settings frames) and ``burst`` (look-alikes inside
one category). Outputs: ``hdr/`` fusions, ``pano/`` stitches, then ``stacks/`` night stacks,
one at a time. A failed output is a ``skipped.csv`` row and never touches a member photo.

Resume marker: a group output counts as done only when the output file exists AND the
ledger holds ``put_score(<group key sha>, "stack:v1", 1)``, written after the output lands.
A crash between the two leaves no marker, so the half-written output is redone.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

#: Consecutive frames this many seconds apart or less may belong to one panorama.
PANO_GAP_S = 8.0
PANO_MIN_FRAMES = 3
#: Ledger score name for a finished group output; bump the suffix if outputs change shape.
STACK_MARKER = "stack:v1"
#: Folder name ``sort.CATEGORY_FOLDERS`` gives the night sky (a test pins the match).
NIGHT_FOLDER = "night-sky"
HDR_DIR, PANO_DIR, STACK_DIR = "hdr", "pano", "stacks"
_TIME_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _megapixels(path) -> float:
    """One frame's megapixels from LibRaw's header alone (same peek as ``photos astro``)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
    return (int(sizes.width) * int(sizes.height)) / 1_000_000.0


def _detect_sequences(paths):
    from .astro_sequence import detect_sequences

    return detect_sequences(paths)


def _chunk_sequence(run):
    from .astro_sequence import chunk_sequence

    return chunk_sequence(run)


def _detect_brackets(paths):
    from .bracket import detect_brackets

    return detect_brackets(paths)


def _phash(rgb) -> int:
    from .cull import compute_cull_score

    return compute_cull_score(rgb).phash


def _fuse(paths, out_path) -> None:
    """Mertens-fuse a bracket, then the normal develop and JPEG export (middle frame's grade)."""
    from . import develop as dv

    decoded = dv.fuse_bracket(paths, align=True)
    params, cluster = dv.compute_batch_corrections([paths[len(paths) // 2]])[0]
    dv.export_jpeg(dv.develop(decoded, params, cluster).rgb, out_path)


def _stitch(paths, out_path) -> None:
    from .panorama import stitch_panorama

    stitch_panorama(paths, out_path)


def _prepare(paths, *, decode_workers, output_dir):
    """The ``photos astro`` handler's gate and decode for one chunk (no calibration frames)."""
    from .astro_orchestrator import prepare_frames
    from .preflight import preflight_astro_stack

    workers = preflight_astro_stack(
        output_dir,
        frame_count=len(paths),
        megapixels=_megapixels(paths[0]),
        decode_workers=decode_workers,
        degrade_workers=True,
    )
    return prepare_frames(paths, decode_workers=workers, detect_meteors=False)


def _stack(prepared, out_path):
    from .astro_orchestrator import stack_run

    return stack_run(prepared, out_path)


@dataclass(frozen=True)
class GroupOps:
    """Seams for the slow stages; defaults are the real pipeline calls."""

    detect_sequences: Callable = _detect_sequences
    chunk_sequence: Callable = _chunk_sequence
    detect_brackets: Callable = _detect_brackets
    phash: Callable = _phash
    fuse: Callable = _fuse
    stitch: Callable = _stitch
    prepare: Callable = _prepare
    stack: Callable = _stack


def real_ops() -> GroupOps:
    return GroupOps()


def capture_datetime(rec) -> Optional[datetime]:
    text = rec.datetime_original
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            stamp = datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
        if rec.subsec and rec.subsec.isdigit():
            stamp += timedelta(seconds=float(f"0.{rec.subsec}"))
        return stamp
    return None


def detect_panorama_runs(records) -> list[tuple[Path, ...]]:
    """Runs of >= ``PANO_MIN_FRAMES`` frames <= ``PANO_GAP_S`` apart with one focal length,
    f-number and exposure time. A false positive costs one failed stitch, never a bad file."""
    stamped = []
    for rec in records:
        stamp = capture_datetime(rec)
        if stamp is not None and None not in (rec.focal_length_mm, rec.f_number, rec.exposure_s):
            stamped.append((stamp, str(rec.path), rec))
    stamped.sort(key=lambda item: item[:2])
    runs, run = [], []

    def close():
        if len(run) >= PANO_MIN_FRAMES:
            runs.append(tuple(r.path for _, r in run))

    for stamp, _, rec in stamped:
        if run:
            last_stamp, last = run[-1]
            same = (rec.focal_length_mm, rec.f_number, rec.exposure_s) == (
                last.focal_length_mm,
                last.f_number,
                last.exposure_s,
            )
            if same and (stamp - last_stamp).total_seconds() <= PANO_GAP_S:
                run.append((stamp, rec))
                continue
            close()
        run = [(stamp, rec)]
    close()
    return runs


def _components(hashes: dict[str, int], threshold: int) -> list[list[str]]:
    """Connected look-alike sets (2+) from ``cull.group_near_duplicates``."""
    from .cull import group_near_duplicates

    near = group_near_duplicates(hashes)
    seen, out = set(), []
    for key in hashes:
        if key in seen or not near[key]:
            continue
        stack, comp = [key], []
        while stack:
            k = stack.pop()
            if k not in seen:
                seen.add(k)
                comp.append(k)
                stack.extend(near[k])
        out.append(comp)
    return out


def _score(row) -> float:
    try:
        return float(row.get("quality_score", ""))
    except (TypeError, ValueError):
        return -1.0


def _best_key(row):
    return (-_score(row), int(row["rank"]), row["category"])


def best_by_group(rows) -> dict[str, dict]:
    """Group id -> its best-scoring member row (rank, then category, break ties)."""
    best: dict[str, dict] = {}
    for row in rows:
        gid = row.get("group_id")
        if gid and (gid not in best or _best_key(row) < _best_key(best[gid])):
            best[gid] = row
    return best


def tag_groups(rows, records, ops: GroupOps, hashes) -> None:
    """Set ``group_id`` / ``group_best_rank`` on ranked-row dicts (in place)."""
    by_src = {str(r["source"]): r for r in rows}
    claimed: set[str] = set()
    counts: Counter = Counter()

    def claim(kind, members):
        members = [m for m in members if m in by_src and m not in claimed]
        if len(members) < 2:
            return
        counts[kind] += 1
        for m in members:
            by_src[m]["group_id"] = f"{kind}-{counts[kind]}"
            claimed.add(m)

    night = [Path(s) for s, r in by_src.items() if r["category"] == NIGHT_FOLDER]
    for run in ops.detect_sequences(night):
        for chunk in ops.chunk_sequence(run):
            claim("night", [str(p) for p in chunk])
    free = [Path(s) for s in by_src if s not in claimed]
    for bracket in ops.detect_brackets(free):
        claim("bracket", [str(p) for p in bracket])
    free = [records[p] for p in (Path(s) for s in by_src if s not in claimed) if p in records]
    for run in detect_panorama_runs(free):
        claim("pano", [str(p) for p in run])
    from .cull import NEAR_DUPLICATE_HAMMING_THRESHOLD

    for category in dict.fromkeys(r["category"] for r in rows):
        pool = {
            s: hashes[s]
            for s, r in by_src.items()
            if r["category"] == category and s not in claimed and s in hashes
        }
        for comp in _components(pool, NEAR_DUPLICATE_HAMMING_THRESHOLD):
            claim("burst", sorted(comp))
    for gid, best in best_by_group(rows).items():
        for row in rows:
            if row.get("group_id") == gid:
                row["group_best_rank"] = int(best["rank"])


@dataclass
class Group:
    id: str
    kind: str
    members: list[str]
    mean_total: float = 0.0


def groups_from_rows(rows) -> list[Group]:
    """Rebuild groups from the ``group_id`` column (capture order inside each group)."""
    members: dict[str, list] = {}
    for row in rows:
        if row.get("group_id"):
            members.setdefault(row["group_id"], []).append(row)
    groups = []
    for gid, items in members.items():
        items.sort(key=lambda r: (r.get("capture_time", ""), str(r["source"])))
        mean = sum(max(_score(r), 0.0) for r in items) / len(items)
        groups.append(Group(gid, gid.rsplit("-", 1)[0], [str(r["source"]) for r in items], mean))
    return groups


def group_key(kind: str, members) -> str:
    return hashlib.sha256("\0".join([kind, *sorted(members)]).encode()).hexdigest()


@dataclass
class Output:
    kind: str  # "hdr" | "pano" | "stacks"
    group_id: str
    rel: str  # path relative to the run folder (the image the page shows)


@dataclass
class StageResult:
    outputs: list[Output] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def build_outputs(
    run_dir: Path, rows, ops: GroupOps, ledger, *, workers: int, no_stacks: bool, on_output
) -> StageResult:
    """HDR, panoramas, then night stacks (best session first); one output at a time."""
    result = StageResult()
    groups = groups_from_rows(rows)
    queue = [g for g in groups if g.kind == "bracket"] + [g for g in groups if g.kind == "pano"]
    queue += sorted((g for g in groups if g.kind == "night"), key=lambda g: -g.mean_total)
    result.stats.update(tagged=len(groups))
    for group in queue:
        if group.kind == "night" and no_stacks:
            continue
        paths = [Path(m) for m in group.members]
        folder = {"bracket": HDR_DIR, "pano": PANO_DIR, "night": STACK_DIR}[group.kind]
        stem = group.id if group.kind == "night" else f"{group.id}-{paths[0].stem}"
        main = run_dir / folder / (f"{stem}.tif" if group.kind == "night" else f"{stem}.jpg")
        shown = main.with_suffix(".jpg")
        key = group_key(group.kind, group.members)
        if main.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
            result.stats["resumed"] += 1
        else:
            main.parent.mkdir(parents=True, exist_ok=True)
            try:
                _produce(group, ops, paths, main, shown, workers, run_dir / folder)
            except Exception as exc:  # a group output must never abort the run
                result.skipped.append((f"group:{group.id}", f"{group.kind} failed: {exc}"))
                result.stats["failed"] += 1
                for leftover in (main, shown, main.with_suffix(".preview.jpg")):
                    leftover.unlink(missing_ok=True)
                continue
            ledger.put_score(key, STACK_MARKER, 1)
            result.stats["built"] += 1
        rel = (shown if shown.is_file() else main).relative_to(run_dir).as_posix()
        result.outputs.append(Output({"night": "stacks"}.get(group.kind, folder), group.id, rel))
        on_output(result.outputs)
    return result


def _produce(group, ops, paths, main, shown, workers, out_dir) -> None:
    if group.kind == "bracket":
        ops.fuse(paths, main)
    elif group.kind == "pano":
        ops.stitch(paths, main)
    else:
        prepared = ops.prepare(paths, decode_workers=workers, output_dir=out_dir)
        report = ops.stack(prepared, main)
        preview = getattr(report, "preview_path", None)
        if preview is not None and Path(preview).is_file():
            os.replace(preview, shown)
