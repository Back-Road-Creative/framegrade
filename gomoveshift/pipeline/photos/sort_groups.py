"""Scene groups for ``photos sort``: tags on ``ranked.csv`` plus HDR, panorama and night-stack outputs.

A group is a tag: every member stays in its own category at its own rank. Kinds, in the
order a shot is claimed: ``night`` (an intervalometer chunk), ``bracket`` (exposure
bracket), ``sequence`` (an intervalometer run outside the night sky), ``pano`` (a fast run
of same-settings frames) and ``burst`` (look-alikes inside one category). Outputs: ``hdr/`` fusions, ``pano/`` stitches, then ``stacks/`` night stacks,
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
import shutil
import statistics
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
#: Folder for lens-cap darks (``measure_frame`` says ``cap_on``): calibration frames, never
#: ranked shots. They join no group, and the page shows only their count.
DARKS_FOLDER = "lens-cap-darks"
#: Darks shot this close to a night session's first or last light calibrate that session.
DARKS_WINDOW_S = 15 * 60
#: A session that left no usable lights for a time-lapse: remembered so ``--resume`` does not
#: decode the whole session again to reach the same verdict.
TIMELAPSE_SKIP_MARKER = "timelapse-skip:v1"
#: Fewest frames worth a time-lapse. The night engine plays 24 fps
#: (``astro_timelapse.TIMELAPSE_FPS``), so 24 frames is one second of video: below that the
#: clip is a flash, not motion, and the stack already shows the scene.
TIMELAPSE_MIN_FRAMES = 24
#: A session too short for a star trail, remembered like ``TIMELAPSE_SKIP_MARKER``.
TRAILS_SKIP_MARKER = "trails-skip:v1"
#: A group output that raised, under its ``group_key``; ``--resort`` does not retry it.
FAILED_MARKER = "stack-failed:v1"
# ``--resort`` only re-files photos; a group output it would have to render waits for --resume.
UNBUILT_REASON = "not built: --resort never renders; a run with --resume builds it"
#: Fewest lights worth a star trail: 24 frames at the measured 32 s interval
#: (``astro_trails``) is about 13 minutes, about 3 degrees of sky rotation. Shorter arcs read
#: as dots, and the stack already shows that scene.
TRAILS_MIN_FRAMES = 24
HDR_DIR, PANO_DIR, STACK_DIR, TIMELAPSE_DIR = "hdr", "pano", "stacks", "timelapse"
TRAILS_DIR = "trails"
#: An intervalometer run outside the night sky (a dawn or cloud sequence): at least
#: ``TIMELAPSE_MIN_FRAMES`` frames of one category, each at most ``SEQUENCE_GAP_S`` after the
#: last, with most gaps (``SEQUENCE_STEADY_SHARE``) within ``SEQUENCE_CADENCE_SLACK`` of the
#: median gap. A timer is steady to the second; a hand-held session is not. JP's 2026-09-22
#: dawn run: 100 frames 18 s apart.
# Hand-held look-alikes taken seconds apart differ by ~6-16 pHash bits; unrelated
# shots a minute or more apart start at ~24 (measured on a real 11-day card, 2026-11-11).
# The 5-bit near-duplicate cut alone never joined them, so a loose match needs both.
BURST_HAMMING_THRESHOLD = 16
BURST_WINDOW_S = 60.0
SEQUENCE_GAP_S = 60.0
SEQUENCE_CADENCE_SLACK = 0.1  # of the median gap, at least 1 s
SEQUENCE_STEADY_SHARE = 0.8
#: Wildlife needs a subject in its pixels, not just a long lens: the share of
#: ``DETAIL_BLOCK_PX`` blocks (frame at ``DETAIL_LONG_EDGE`` px) whose Laplacian variance
#: exceeds ``DETAIL_VARIANCE_MIN``. Measured on the camera's embedded previews of JP's
#: 2026-11-09 card: deer and turkeys 0.91-1.00, a 250 mm cloud 0.08, a 210 mm haze 0.14.
#: Below the floor the frame is filed landscape. Known limit: a lone bird against open sky
#: fills too little of the frame and reads landscape too.
WILDLIFE_DETAIL_FLOOR = 0.5
DETAIL_LONG_EDGE = 1024
DETAIL_BLOCK_PX = 32
DETAIL_VARIANCE_MIN = 50.0
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

    # The chunk's linear artifact (beside its master) is what the session combine reads.
    return stack_run(prepared, out_path, linear_artifact_path=Path(out_path).with_suffix(".linear"))


def _combine_session(linears, out_tif, out_jpg) -> None:
    """The deep combine ``photos astro`` runs over chunk artifacts, plus its browser JPEG."""
    from .night_products import _render_deep, render_master_preview

    _render_deep([Path(p) for p in linears], Path(out_tif))
    render_master_preview(Path(out_tif), Path(out_jpg))


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


def _subject_detail(path) -> float:
    """Share (0..1) of the frame carrying fine detail (see ``WILDLIFE_DETAIL_FLOOR``). A raw
    is read through its embedded JPEG preview, so nothing is developed; raises when there
    is nothing to read."""
    import io

    import cv2
    import numpy as np
    from PIL import Image

    from .dedupe import RAW_EXTENSIONS

    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            raise ValueError(f"{path}: no embedded JPEG preview")
        source = io.BytesIO(thumb.data)
    else:
        source = path
    with Image.open(source) as img:
        img.draft("L", (DETAIL_LONG_EDGE, DETAIL_LONG_EDGE))
        gray = np.asarray(img.convert("L"))
    scale = DETAIL_LONG_EDGE / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    b = DETAIL_BLOCK_PX
    h, w = lap.shape[0] // b * b, lap.shape[1] // b * b
    if not h or not w:
        raise ValueError(f"{path}: frame smaller than one {b}-px block")
    blocks = lap[:h, :w].reshape(h // b, b, w // b, b).var(axis=(1, 3))
    return float(np.mean(blocks > DETAIL_VARIANCE_MIN))


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
    #: ``(chunk_linear_dirs, out_tif, out_jpg)``: deep-combines one night session's chunks.
    combine_session: Callable = _combine_session
    #: ``(paths) -> (lights, lens_cap_darks)`` for one night session.
    classify_night: Callable = _classify_night
    #: ``(path) -> 0..1`` share of the frame carrying fine detail; wildlife needs a subject.
    subject_detail: Callable = _subject_detail
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


def _components(hashes: dict[str, int], when: dict[str, datetime]) -> list[list[str]]:
    """Connected look-alike sets (2+): a near-duplicate pHash at any time apart, or a
    looser ``BURST_HAMMING_THRESHOLD`` match taken within ``BURST_WINDOW_S`` of each other."""
    from .cull import NEAR_DUPLICATE_HAMMING_THRESHOLD, hamming_distance

    keys = list(hashes)
    near: dict[str, list[str]] = {k: [] for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            bits = hamming_distance(hashes[a], hashes[b])
            close = (
                a in when
                and b in when
                and abs((when[a] - when[b]).total_seconds()) <= BURST_WINDOW_S
            )
            if bits <= NEAR_DUPLICATE_HAMMING_THRESHOLD or (
                close and bits <= BURST_HAMMING_THRESHOLD
            ):
                near[a].append(b)
                near[b].append(a)
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


def detect_interval_runs(records) -> list[list[Path]]:
    """Runs of frames shot on a timer (see ``SEQUENCE_GAP_S``), each in capture order."""
    timed = sorted((t, r.path) for r in records if (t := capture_datetime(r)) is not None)
    runs: list[list] = []
    for when, path in timed:
        if runs and (when - runs[-1][-1][0]).total_seconds() <= SEQUENCE_GAP_S:
            runs[-1].append((when, path))
        else:
            runs.append([(when, path)])
    steady = []
    for run in runs:
        if len(run) < TIMELAPSE_MIN_FRAMES:
            continue
        gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(run, run[1:])]
        mid = statistics.median(gaps)
        slack = max(1.0, SEQUENCE_CADENCE_SLACK * mid)
        if sum(abs(g - mid) <= slack for g in gaps) >= SEQUENCE_STEADY_SHARE * len(gaps):
            steady.append([path for _when, path in run])
    return steady


def tag_groups(rows, records, ops: GroupOps, hashes) -> None:
    """Set ``group_id`` / ``group_best_rank`` on ranked-row dicts (in place)."""
    by_src = {str(r["source"]): r for r in rows}
    claimed = {s for s, r in by_src.items() if r["category"] == DARKS_FOLDER}  # never grouped
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
    for category in dict.fromkeys(r["category"] for r in rows):
        if category == NIGHT_FOLDER:  # night runs are claimed above, chunked for stacking
            continue
        pool = [
            records[p]
            for p in (Path(s) for s, r in by_src.items() if r["category"] == category)
            if p in records and str(p) not in claimed
        ]
        for run in detect_interval_runs(pool):
            claim("sequence", [str(p) for p in run])
    free = [records[p] for p in (Path(s) for s in by_src if s not in claimed) if p in records]
    for run in detect_panorama_runs(free):
        claim("pano", [str(p) for p in run])
    when = {
        s: t
        for s in by_src
        if Path(s) in records and (t := capture_datetime(records[Path(s)])) is not None
    }
    for category in dict.fromkeys(r["category"] for r in rows):
        pool = {
            s: hashes[s]
            for s, r in by_src.items()
            if r["category"] == category and s not in claimed and s in hashes
        }
        for comp in _components(pool, when):
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
    run_dir: Path,
    rows,
    ops: GroupOps,
    ledger,
    *,
    workers: int,
    no_stacks: bool,
    on_output,
    triaged=frozenset(),
    build=True,
) -> StageResult:
    """HDR, panoramas, night stacks (best session first), then star trails and time-lapses;
    one at a time. ``triaged``: sources this run already cleared as lights (not lens-cap
    darks), so a session made only of them is not decoded a second time. Without ``build``
    (``--resort``) nothing is rendered or decoded: a finished output whose members are
    unchanged stays on the page, and every other group becomes a ``skipped.csv`` row
    (``UNBUILT_REASON``, or the earlier failure) that ``--resume`` builds."""
    result = StageResult()
    groups = groups_from_rows(rows)
    queue = [g for g in groups if g.kind == "bracket"] + [g for g in groups if g.kind == "pano"]
    queue += sorted((g for g in groups if g.kind == "night"), key=lambda g: -g.mean_total)
    result.stats.update(tagged=len(groups))
    by_src = {str(r["source"]): r for r in rows}
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
        elif group.kind == "night" and _starless(group, by_src):
            result.skipped.append(
                (f"group:{group.id}", f"bright sky: no stars measured in {len(paths)} frames")
            )
            result.stats["starless"] += 1
            continue
        elif not build:
            _unbuilt(
                result, f"group:{group.id}", ledger.get_score(key, FAILED_MARKER) == 1, group.kind
            )
            continue
        else:
            main.parent.mkdir(parents=True, exist_ok=True)
            try:
                _produce(group, ops, paths, main, shown, workers, run_dir / folder)
            except Exception as exc:  # a group output must never abort the run
                result.skipped.append((f"group:{group.id}", f"{group.kind} failed: {exc}"))
                result.stats["failed"] += 1
                ledger.put_score(key, FAILED_MARKER, 1)
                for leftover in (main, shown, main.with_suffix(".preview.jpg")):
                    leftover.unlink(missing_ok=True)
                continue
            ledger.put_score(key, STACK_MARKER, 1)
            result.stats["built"] += 1
        rel = (shown if shown.is_file() else main).relative_to(run_dir).as_posix()
        result.outputs.append(Output({"night": "stacks"}.get(group.kind, folder), group.id, rel))
        on_output(result.outputs)
    if not no_stacks:
        _combine_sessions(run_dir, groups, ops, ledger, result, on_output, build)
    darks = sorted(
        (stamp, str(r["source"]))
        for r in rows
        if r["category"] == DARKS_FOLDER and (stamp := _parse_stamp(r.get("capture_time")))
    )
    _build_timelapses(
        run_dir, groups, ops, ledger, result, no_stacks, on_output, darks, triaged, build
    )
    return result


def _unbuilt(result, path, failed=False, kind="") -> None:
    """A ``--resort`` row for an output it may not render; ``--resume`` builds it."""
    if failed:
        result.skipped.append((path, f"{kind} failed in an earlier run; --resume retries it"))
        result.stats["failed"] += 1
    else:
        result.skipped.append((path, UNBUILT_REASON))
        result.stats["unbuilt"] += 1


def _starless(group, by_src) -> bool:
    """Every frame of a night chunk measured ``technical.usable`` 0 (a moonlit or twilight
    sky with no stars to register), so stacking it would only fail after a long decode. A
    frame with no measurement keeps the chunk."""
    marks = [by_src.get(m, {}).get("technical.usable") for m in group.members]
    return bool(marks) and all(m not in (None, "") and float(m) == 0 for m in marks)


def _combine_sessions(run_dir, groups, ops, ledger, result, on_output, build=True) -> None:
    """One page result per night session of two or more built chunks: the deep combine of
    their linear artifacts, ``stacks/<first-id>-session.tif``. The chunk stacks stay on disk
    as intermediates and leave the page; their ``.linear`` inputs are removed once the
    session result is built. A failed combine keeps only the session's best chunk (highest
    mean total) on the page, and keeps the inputs so a resume can retry."""
    for session in night_sessions(groups):
        shown = {o.group_id: o for o in result.outputs if o.kind == "stacks"}
        built = [g for g in session if g.id in shown]
        if len(built) < 2:
            continue
        sid = session[0].id
        main = run_dir / STACK_DIR / f"{sid}-session.tif"
        jpg = main.with_suffix(".jpg")
        key = group_key("session", [m for g in built for m in g.members])
        if main.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
            result.stats["resumed"] += 1
            keep = Output(
                "stacks", sid, (jpg if jpg.is_file() else main).relative_to(run_dir).as_posix()
            )
        elif not build:
            _unbuilt(result, f"group:{sid}-session")
            continue
        else:
            try:
                ops.combine_session(
                    [run_dir / STACK_DIR / f"{g.id}.linear" for g in built], main, jpg
                )
            except Exception as exc:  # a session result must never abort the run
                chunks = ", ".join(g.id for g in built)
                result.skipped.append(
                    (f"group:{sid}-session", f"session combine failed ({chunks}): {exc}")
                )
                result.stats["failed"] += 1
                for leftover in (main, jpg, main.with_suffix(".preview.jpg")):
                    leftover.unlink(missing_ok=True)
                keep = shown[max(built, key=lambda g: g.mean_total).id]
            else:
                ledger.put_score(key, STACK_MARKER, 1)
                result.stats["built"] += 1
                for g in built:  # spent combine inputs, ~0.7 GB each at 24 MP
                    shutil.rmtree(run_dir / STACK_DIR / f"{g.id}.linear", ignore_errors=True)
                keep = Output(
                    "stacks", sid, (jpg if jpg.is_file() else main).relative_to(run_dir).as_posix()
                )
        drop = {id(shown[g.id]) for g in built}
        result.outputs[:] = [o for o in result.outputs if id(o) not in drop] + [keep]
        on_output(result.outputs)


def _session_darks(darks, captured) -> list[Path]:
    """The ``(stamp, source)`` darks shot within ``DARKS_WINDOW_S`` of a session's lights."""
    stamps = [t for t in map(_parse_stamp, captured) if t]
    if not stamps:
        return []
    window = timedelta(seconds=DARKS_WINDOW_S)
    lo, hi = min(stamps) - window, max(stamps) + window
    return [Path(s) for t, s in darks if lo <= t <= hi]


def _build_timelapses(
    run_dir,
    groups,
    ops,
    ledger,
    result,
    no_stacks,
    on_output,
    darks=(),
    triaged=frozenset(),
    build=True,
) -> None:
    """Per night session (not under ``--no-stacks``) a star trail then a time-lapse, sharing
    one light/dark triage plus the session's ``DARKS_FOLDER`` darks; per long burst a
    time-lapse."""
    units = []  # (id, kind, member sources, developed JPEGs, nearby darks)
    if not no_stacks:
        for session in night_sessions(groups):
            members = [m for g in session for m in g.members]
            if len(members) >= TIMELAPSE_MIN_FRAMES:
                near = _session_darks(darks, [c for g in session for c in g.captured])
                units.append((session[0].id, "night", members, [], near))
    for g in groups:
        if g.kind in ("burst", "sequence") and len(g.members) >= TIMELAPSE_MIN_FRAMES:
            units.append((g.id, "burst", g.members, [o for o in g.outputs if o], []))
    for gid, kind, members, jpegs, near in units:  # kind: "night" | "burst" | "sequence"
        triage: list = []  # the session's (lights, darks), decoded at most once

        def classify(members=members, triage=triage, near=near):
            if not triage:
                if all(m in triaged for m in members):  # this run already checked them
                    lights, found = [Path(m) for m in members], []
                else:
                    lights, found = ops.classify_night([Path(m) for m in members])
                found = list(found)
                triage.append((lights, found + [d for d in near if d not in found]))
            return triage[0]

        if kind == "night":
            _build_trail(run_dir, gid, members, classify, ops, ledger, result, on_output, build)
        mp4 = run_dir / TIMELAPSE_DIR / f"{gid}.mp4"
        poster = mp4.with_suffix(".jpg")
        key = group_key("timelapse", members)
        if mp4.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
            result.stats["resumed"] += 1
        elif ledger.get_score(key, TIMELAPSE_SKIP_MARKER) == 1:
            continue
        elif not build:
            _unbuilt(result, f"group:{gid}-timelapse")
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


def _build_trail(
    run_dir, gid, members, classify, ops, ledger, result, on_output, build=True
) -> None:
    """One star trail for a night session: resume, skip and failure as a time-lapse."""
    tif = run_dir / TRAILS_DIR / f"{gid}.tif"
    jpg = tif.with_suffix(".jpg")
    key = group_key("trails", members)
    if tif.is_file() and jpg.is_file() and ledger.get_score(key, STACK_MARKER) == 1:
        result.stats["resumed"] += 1
    elif ledger.get_score(key, TRAILS_SKIP_MARKER) == 1:
        return
    elif not build:
        _unbuilt(result, f"group:{gid}-trails")
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
