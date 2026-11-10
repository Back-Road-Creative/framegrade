"""Scene groups for ``photos sort``: tags on ``ranked.csv`` plus HDR, panorama and night-stack outputs.

A group is a tag: every member stays in its own category at its own rank. Kinds, in the
order a shot is claimed: ``night`` (an intervalometer chunk), ``bracket`` (exposure
bracket), ``pano`` (a fast run of same-settings frames) and ``burst`` (look-alikes inside
one category). Outputs: ``hdr/`` fusions, ``pano/`` stitches, then ``stacks/`` night stacks,
one at a time. Then, per night session, ``trails/``: one star-trail TIFF master plus a
same-stem JPEG; and ``timelapse/``: one MP4 (plus a same-stem poster JPEG) per night session
and per long burst. A failed output is a ``skipped.csv`` row and never touches a member photo.

Resume marker: a group output counts as done only when the output file exists AND the
ledger holds ``put_score(<group key sha>, "stack:v1", 1)``, written after the output lands.
A crash between the two leaves no marker, so the half-written output is redone. A time-lapse
uses the same marker under its own key (``group_key("timelapse", ...)``); a star trail under
``group_key("trails", ...)``.
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
#: A session that left no usable lights for a time-lapse: remembered so ``--resume`` does not
#: decode the whole session again to reach the same verdict.
TIMELAPSE_SKIP_MARKER = "timelapse-skip:v1"
#: Fewest frames worth a time-lapse. The night engine plays 24 fps
#: (``astro_timelapse.TIMELAPSE_FPS``), so 24 frames is one second of video: below that the
#: clip is a flash, not motion, and the stack already shows the scene.
TIMELAPSE_MIN_FRAMES = 24
#: A session too short for a star trail, remembered like ``TIMELAPSE_SKIP_MARKER``.
TRAILS_SKIP_MARKER = "trails-skip:v1"
#: Fewest lights worth a star trail: 24 frames at the measured 32 s interval
#: (``astro_trails``) is about 13 minutes, about 3 degrees of sky rotation. Shorter arcs read
#: as dots, and the stack already shows that scene.
TRAILS_MIN_FRAMES = 24
HDR_DIR, PANO_DIR, STACK_DIR, TIMELAPSE_DIR = "hdr", "pano", "stacks", "timelapse"
TRAILS_DIR = "trails"
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


def _classify_night(paths):
    """Split a night session into ``(lights, lens_cap_darks)`` with the stack's own triage
    verdict: ``measure_frame``'s ``exposure_class`` (``astro_triage.classify_exposure``, #289)."""
    from .astro_orchestrator import _luma, decode_with_he_fallback
    from .astro_triage import measure_frame

    lights, darks = [], []
    for path in paths:
        decoded, _ = decode_with_he_fallback(path, None, None)
        if decoded is None:
            lights.append(path)  # the renderer names an undecodable frame; it is not a dark
            continue
        metric = measure_frame(path, _luma(decoded.rgb), detect_meteors=False)
        (darks if metric.exposure_class == "cap_on" else lights).append(path)
    return lights, darks


def _poster(video, poster, frames, fps) -> None:
    """The video's middle frame as a JPEG poster (at most 1080 px tall), via ffmpeg."""
    import subprocess

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(frames, 1) / fps / 2:.3f}"]
    cmd += ["-i", str(video), "-frames:v", "1", "-vf", "scale=-2:'min(1080,ih)'", str(poster)]
    done = subprocess.run(cmd, capture_output=True, timeout=300)
    if done.returncode != 0 or not Path(poster).is_file():
        tail = done.stderr.decode(errors="replace").strip()[-300:]
        raise RuntimeError(f"poster frame failed: exit {done.returncode}: {tail}")


def _validated_darks(lights, darks) -> list:
    """``darks`` when they validate against the lights (same identity check as a stack); a
    set that does not match is dropped, never subtracted blind."""
    from .astro_calibrate import AstroCalibrationError
    from .astro_orchestrator import build_validated_master_dark

    if not darks:
        return []
    try:
        build_validated_master_dark(lights[:1], darks)
    except AstroCalibrationError:
        return []
    return list(darks)


def _night_timelapse(lights, darks, out_mp4, out_jpg) -> None:
    """One session's lights to an MP4 with the night product's fps and height cap; darks
    pass through ``_validated_darks``."""
    from .astro_timelapse import (
        TIMELAPSE_FPS,
        TIMELAPSE_HEIGHT_DEFAULT,
        preflight_timelapse,
        render_timelapse,
    )

    darks = _validated_darks(lights, darks)
    preflight_timelapse(out_mp4.parent, megapixels=_megapixels(lights[0]), dark_count=len(darks))
    video, used, _rejected = render_timelapse(
        lights, out_mp4, darks, fps=TIMELAPSE_FPS, scale_height=TIMELAPSE_HEIGHT_DEFAULT
    )
    _poster(video, out_jpg, used, TIMELAPSE_FPS)


def _star_trails(lights, darks, out_tif, out_jpg) -> None:
    """One session's lights to a star-trail master (``astro_trails.render_trail``: unaligned
    per-pixel maximum) plus a browser JPEG from that master, as the night products export it.
    Darks pass through ``_validated_darks``; a maximum keeps a hot pixel forever, so an
    unmatched set is never subtracted."""
    from .astro_trails import preflight_trail, render_trail
    from .night_products import render_master_preview

    darks = _validated_darks(lights, darks)
    preflight_trail(out_tif.parent, megapixels=_megapixels(lights[0]), dark_count=len(darks))
    master, _used, _rejected = render_trail(lights, out_tif, darks)
    render_master_preview(Path(master), out_jpg)


def _still_timelapse(jpegs, out_mp4, out_jpg) -> None:
    """A burst's developed JPEGs to an MP4 through the shared still-frame core. The core
    orders stills by file name and the developed names are rank-first, so the frames go in
    under sequence-numbered links to keep capture order."""
    import shutil
    import tempfile

    from ..timelapse import TIMELAPSE_FPS, render_stills_timelapse
    from .astro_timelapse import TIMELAPSE_HEIGHT_DEFAULT

    with tempfile.TemporaryDirectory(prefix="sort-lapse-") as tmp:
        ordered = []
        for i, jpeg in enumerate(jpegs):
            link = Path(tmp) / f"{i:06d}.jpg"
            try:
                os.symlink(Path(jpeg).resolve(), link)
            except OSError:
                shutil.copyfile(jpeg, link)
            ordered.append(link)
        video, used, _rejected = render_stills_timelapse(
            ordered, out_mp4, fps=TIMELAPSE_FPS, scale_height=TIMELAPSE_HEIGHT_DEFAULT
        )
    _poster(video, out_jpg, used, TIMELAPSE_FPS)


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
    #: ``(paths) -> (lights, lens_cap_darks)`` for one night session.
    classify_night: Callable = _classify_night
    #: ``(lights, darks, out_mp4, out_jpg)``: writes the video and its poster.
    night_timelapse: Callable = _night_timelapse
    #: ``(developed_jpegs_in_capture_order, out_mp4, out_jpg)``: writes the video and poster.
    still_timelapse: Callable = _still_timelapse
    #: ``(lights, darks, out_tif, out_jpg)``: writes the star-trail master and its JPEG.
    star_trails: Callable = _star_trails


def real_ops() -> GroupOps:
    return GroupOps()


def _parse_stamp(text) -> Optional[datetime]:
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def capture_datetime(rec) -> Optional[datetime]:
    stamp = _parse_stamp(rec.datetime_original)
    if stamp is not None and rec.subsec and rec.subsec.isdigit():
        stamp += timedelta(seconds=float(f"0.{rec.subsec}"))
    return stamp


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
    outputs: list[str] = field(default_factory=list)  # each member's developed JPEG
    captured: list[str] = field(default_factory=list)  # each member's capture_time text


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
        sources = [str(r["source"]) for r in items]
        outputs = [str(r.get("output") or "") for r in items]
        captured = [str(r.get("capture_time") or "") for r in items]
        groups.append(Group(gid, gid.rsplit("-", 1)[0], sources, mean, outputs, captured))
    return groups


def night_sessions(groups) -> list[list[Group]]:
    """Night groups joined into sessions: chunks of one ``detect_sequences`` run. A run holds
    frames at most ``MAX_FRAME_GAP_S`` apart, so a chunk that starts inside that gap of the
    previous chunk's last frame continues its session. Without capture times the chunks stay
    apart. Rebuilt from the rows alone, so ``--resume`` finds the same sessions."""
    from .astro_sequence import MAX_FRAME_GAP_S

    night = sorted(
        (g for g in groups if g.kind == "night"), key=lambda g: int(g.id.rsplit("-", 1)[1])
    )
    sessions: list[list[Group]] = []
    for group in night:
        last = _parse_stamp(sessions[-1][-1].captured[-1]) if sessions else None
        first = _parse_stamp(group.captured[0]) if group.captured else None
        if last and first and 0 <= (first - last).total_seconds() <= MAX_FRAME_GAP_S:
            sessions[-1].append(group)
        else:
            sessions.append([group])
    return sessions


def group_key(kind: str, members) -> str:
    return hashlib.sha256("\0".join([kind, *sorted(members)]).encode()).hexdigest()


@dataclass
class Output:
    kind: str  # "hdr" | "pano" | "stacks" | "trails" | "timelapse"
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
    """HDR, panoramas, night stacks (best session first), then star trails and time-lapses;
    one at a time."""
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
    _build_timelapses(run_dir, groups, ops, ledger, result, no_stacks, on_output)
    return result


def _build_timelapses(run_dir, groups, ops, ledger, result, no_stacks, on_output) -> None:
    """Per night session (not under ``--no-stacks``) a star trail then a time-lapse, sharing
    one light/dark triage; per long burst a time-lapse."""
    units = []  # (id, kind, member sources, developed JPEGs)
    if not no_stacks:
        for session in night_sessions(groups):
            members = [m for g in session for m in g.members]
            if len(members) >= TIMELAPSE_MIN_FRAMES:
                units.append((session[0].id, "night", members, []))
    for g in groups:
        if g.kind == "burst" and len(g.members) >= TIMELAPSE_MIN_FRAMES:
            units.append((g.id, "burst", g.members, [o for o in g.outputs if o]))
    for gid, kind, members, jpegs in units:
        triage: list = []  # the session's (lights, darks), decoded at most once

        def classify(members=members, triage=triage):
            if not triage:
                triage.append(ops.classify_night([Path(m) for m in members]))
            return triage[0]

        if kind == "night":
            _build_trail(run_dir, gid, members, classify, ops, ledger, result, on_output)
        mp4 = run_dir / TIMELAPSE_DIR / f"{gid}.mp4"
        poster = mp4.with_suffix(".jpg")
        key = group_key("timelapse", members)
        if mp4.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
            result.stats["resumed"] += 1
        elif ledger.get_score(key, TIMELAPSE_SKIP_MARKER) == 1:
            continue
        else:
            try:
                if kind == "night":
                    lights, darks = classify()
                    if len(lights) < TIMELAPSE_MIN_FRAMES:
                        ledger.put_score(key, TIMELAPSE_SKIP_MARKER, 1)
                        continue
                    mp4.parent.mkdir(parents=True, exist_ok=True)
                    ops.night_timelapse(lights, darks, mp4, poster)
                elif len(jpegs) >= TIMELAPSE_MIN_FRAMES:
                    mp4.parent.mkdir(parents=True, exist_ok=True)
                    ops.still_timelapse([Path(j) for j in jpegs], mp4, poster)
                else:
                    continue
            except Exception as exc:  # a time-lapse must never abort the run
                result.skipped.append((f"group:{gid}", f"timelapse failed: {exc}"))
                result.stats["failed"] += 1
                for leftover in (mp4, poster):
                    leftover.unlink(missing_ok=True)
                continue
            ledger.put_score(key, STACK_MARKER, 1)
            result.stats["built"] += 1
        result.outputs.append(Output("timelapse", gid, f"{TIMELAPSE_DIR}/{gid}.mp4"))
        on_output(result.outputs)


def _build_trail(run_dir, gid, members, classify, ops, ledger, result, on_output) -> None:
    """One star trail for a night session: resume, skip and failure as a time-lapse."""
    tif = run_dir / TRAILS_DIR / f"{gid}.tif"
    jpg = tif.with_suffix(".jpg")
    key = group_key("trails", members)
    if tif.is_file() and jpg.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
        result.stats["resumed"] += 1
    elif ledger.get_score(key, TRAILS_SKIP_MARKER) == 1:
        return
    else:
        try:
            lights, darks = classify()
            if len(lights) < TRAILS_MIN_FRAMES:
                ledger.put_score(key, TRAILS_SKIP_MARKER, 1)
                return
            tif.parent.mkdir(parents=True, exist_ok=True)
            ops.star_trails(lights, darks, tif, jpg)
        except Exception as exc:  # a star trail must never abort the run
            result.skipped.append((f"group:{gid}", f"trails failed: {exc}"))
            result.stats["failed"] += 1
            for leftover in (tif, jpg):
                leftover.unlink(missing_ok=True)
            return
        ledger.put_score(key, STACK_MARKER, 1)
        result.stats["built"] += 1
    result.outputs.append(Output("trails", gid, f"{TRAILS_DIR}/{gid}.jpg"))
    on_output(result.outputs)


def _produce(group, ops, paths, main, shown, workers, out_dir) -> None:
    if group.kind == "bracket":
        ops.fuse(paths, main)
    elif group.kind == "pano":
        # The stitcher reads with cv2.imread, which cannot open a RAW: stitch the
        # developed JPEGs this run wrote (graded and upright, as the page shows them).
        ops.stitch([Path(o) for o in group.outputs], main)
    else:
        prepared = ops.prepare(paths, decode_workers=workers, output_dir=out_dir)
        report = ops.stack(prepared, main)
        preview = getattr(report, "preview_path", None)
        if preview is not None and Path(preview).is_file():
            os.replace(preview, shown)
