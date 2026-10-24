"""Insert engine: key-cut stills and clips into a rendered long-form.

``plan_inserts`` decides where each insert cuts in and out; ``apply_inserts``
stream-copies everything outside the bounding GOPs and re-encodes only
``[cut_in, cut_out)``. ``anchor_photo_insert`` places a photo from its capture
clock against the day's track; ``InsertGate`` refuses an insert captured
outside the LF's label counties and runs the blur and GPS-strip gates before
any encode. See ``docs/pipeline/153-insert-engine.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Optional, Sequence

from keycut import probe_keyframes
from PIL import Image

from .blur_gate import BlurGateError, blur_required, master_blur_proven
from .encoding_constants import RENDITIONS
from .geo_constants import US_STATE_ABBREVS
from .montage_builder import (
    GOP_DURATION_SEC,  # noqa: F401  (nominal GOP; the cut uses probed keyframes)
    concat_copy_argv,
    copy_range_argv,
    keyframe_span,
    lf_reencode_args,
)
from .quality import upscale_plan
from .route_story.anchors import MediaEvidence, anchor_media_to_segments, captured_at_route_seconds
from .route_story.metrics import RoutePoint

logger = logging.getLogger(__name__)

STYLES = ("full", "letterbox")
STILL_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic"}


class InsertError(RuntimeError):
    """An insert cannot be planned or applied."""


@dataclass(frozen=True)
class Insert:
    """One insert: ``media`` shown from ``at_s``. A clip ignores ``hold_s``.

    ``capture`` is the ``(lat, lon)`` where the media was taken; the county gate
    reads it (a still with none falls back to its EXIF fix).
    """

    at_s: float
    media: Path
    hold_s: float = 4.0
    style: str = "full"
    capture: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    duration: float


@dataclass(frozen=True)
class InsertPlan:
    insert: Insert
    kind: str  # "still" | "clip"
    cut_in: float
    cut_out: float
    duration: float
    card_path: Optional[Path] = None
    scaler_filter: str = ""
    upscaled_from: Optional[list[int]] = None
    lf_fps: float = 30.0
    lf_duration: float = 0.0


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    stream_copied: bool


@dataclass
class InsertResult:
    output: Path
    spans: list[Span] = field(default_factory=list)
    upscaled: list[list[int]] = field(default_factory=list)


# Accepted anchor statuses: a track time exists and the evidence agrees.
_ANCHORED = {"embedded_coordinates", "timestamp_interpolated"}


def anchor_photo_insert(
    photo: Path,
    *,
    captured_at_utc: Optional[datetime],
    track_started_at_utc: datetime,
    segments: Sequence[Sequence[RoutePoint]],
    lf_start_s: float,
    lf_duration_s: float,
    gps: Optional[tuple[float, float]] = None,
    hold_s: float = 4.0,
    style: str = "full",
) -> Insert:
    """A photo's insert, placed by its capture clock against the day's track.

    ``captured_at_utc`` must be an explicitly zoned instant; it is moved onto
    the track's clock (``track_started_at_utc``) and located with
    ``anchors.anchor_media_to_segments``. ``lf_start_s`` is the LF's start on
    that same clock. A photo the track cannot place (outside the recording, in
    a gap, GPS contradicting the clock) or one outside this LF is refused, never
    guessed. The anchored position becomes ``Insert.capture`` for the county gate.
    """
    if captured_at_utc is None:
        raise InsertError(f"{Path(photo).name}: no capture time, cannot anchor to the track")
    media = MediaEvidence(
        Path(photo).name,
        captured_at_s=captured_at_route_seconds(captured_at_utc, track_started_at_utc),
        latitude=gps[0] if gps else None,
        longitude=gps[1] if gps else None,
    )
    anchor = anchor_media_to_segments(media, segments)
    if anchor.status not in _ANCHORED or anchor.track_time_s is None:
        raise InsertError(f"{media.media_id}: cannot be placed on the track ({anchor.status})")
    at_s = anchor.track_time_s - lf_start_s
    if not 0.0 <= at_s < lf_duration_s:
        raise InsertError(f"{media.media_id}: captured at {at_s:.1f}s, outside this LF")
    return Insert(at_s, Path(photo), hold_s, style, (anchor.latitude, anchor.longitude))


_COUNTY_RE = re.compile(r"([A-Z][^,&]*?(?: & [A-Z][^,&]*?)*) County(?:, ([A-Z]{2})\b)?")


def label_counties(label: Optional[str]) -> frozenset[tuple[str, str]]:
    """``{(county, STATE)}`` named by an LF label; empty when it names none.

    Reads ``"A County, SC"``, ``"A & B County, SC"`` and ``"A County, SC & B
    County, GA"`` (``title_builder._multi_county_label``). Names are lowercase
    without the word County; STATE is the USPS code, or ``""`` when absent.
    """
    found = set()
    for names, state in _COUNTY_RE.findall(label or ""):
        found.update((n.strip().lower(), state) for n in names.split(" & "))
    return frozenset(found)


def _geocoder_county_of(lat: float, lon: float) -> tuple[Optional[str], Optional[str]]:
    from .ingest.geocoding import _get_shared_geocoder

    detail = _get_shared_geocoder().reverse_geocode_detailed(lat, lon) or {}
    return detail.get("county"), detail.get("state")


@dataclass(frozen=True)
class _Gated:
    source: Path


@dataclass(frozen=True)
class InsertGate:
    """Everything an insert must pass before any encode.

    ``label_counties`` comes from the LF's label (``per_lf_locations[idx]`` in
    ``RunSpec``, via :func:`label_counties`). ``blur`` is the stills' privacy
    switch (``None`` = the run's setting, ``False`` = ``--skip-blur``);
    ``clip_blurred`` says whether a clip is provably blurred.
    """

    label_counties: frozenset[tuple[str, str]]
    county_of: Callable[[float, float], tuple[Optional[str], Optional[str]]] = _geocoder_county_of
    blur: Optional[bool] = None
    clip_blurred: Callable[[Path], bool] = lambda p: False  # noqa: E731  (fail closed)
    privacy_gate: Optional[Callable[..., Any]] = None

    @classmethod
    def from_config(cls, config, label: Optional[str], **kw) -> "InsertGate":
        """The gate for one LF under ``config``'s blur policy."""
        return cls(
            label_counties(label),
            blur=None if blur_required(config) else False,
            clip_blurred=lambda p: master_blur_proven(config, p),
            **kw,
        )


def _capture_position(ins: Insert) -> Optional[tuple[float, float]]:
    if ins.capture is not None:
        return ins.capture
    if Path(ins.media).suffix.lower() in STILL_SUFFIXES:
        from .photos.exif_gps import extract_exif_gps

        return extract_exif_gps(ins.media)
    return None


def county_in_label(
    county: str, state: Optional[str], label_counties: frozenset[tuple[str, str]]
) -> bool:
    """Whether a geocoded ``county`` / ``state`` is one of the LF label's counties."""
    key = re.sub(r" county$", "", county.strip().lower())
    st = (state or "").strip()
    st = st.upper() if len(st) == 2 else US_STATE_ABBREVS.get(st.title(), "")
    # Fail closed on state: a county name is only unique within its state. A label
    # pair with a state needs an insert whose state is known and equal; a state-less
    # pair matches on the name alone only when it is the label's sole state-less pair.
    lone_stateless = sum(1 for _, s in label_counties if not s) == 1
    return any(key == c and (st == s if s else lone_stateless) for c, s in label_counties)


def _check_county(ins: Insert, gate: InsertGate) -> None:
    name = Path(ins.media).name
    pos = _capture_position(ins)
    if pos is None:
        raise InsertError(f"{name}: no capture position, cannot check the LF's counties")
    county, state = gate.county_of(*pos)
    if not county:
        raise InsertError(f"{name}: capture county unresolved, refusing")
    if not gate.label_counties:
        raise InsertError(f"{name}: the LF label names no county to check against")
    if not county_in_label(county, state, gate.label_counties):
        label = ", ".join(
            f"{c.title()} County, {s}".rstrip(", ") for c, s in sorted(gate.label_counties)
        )
        raise InsertError(f"{name}: captured in {county}, {state}, outside the LF label ({label})")


def gate_inserts(inserts: list[Insert], gate: InsertGate, *, work_dir: Path) -> list[Insert]:
    """County, then blur and GPS-strip, for every insert; raises on the first miss.

    A still comes back as its privacy-gated working copy (blurred, EXIF/GPS
    stripped, ``route_story.privacy``), so the card is drawn from the copy and
    the original is never read again. A clip must carry the ``.blurred`` marker
    when the run requires blur; its container tags never reach the splice
    (``_encode_argv`` maps metadata from the LF only).
    """
    out = []
    for i, ins in enumerate(inserts):
        _check_county(ins, gate)
        if Path(ins.media).suffix.lower() in STILL_SUFFIXES:
            from .route_story.privacy import apply_privacy_gate

            run = gate.privacy_gate or apply_privacy_gate
            res = run(
                [_Gated(Path(ins.media))], blur=gate.blur, work_dir=work_dir / f"gated_{i:02d}"
            )
            if not res.photos:
                raise InsertError(f"{Path(ins.media).name}: withheld by the privacy gate")
            ins = replace(ins, media=res.photos[0].source)
        elif not gate.clip_blurred(Path(ins.media)):
            raise BlurGateError(
                f"BLUR GATE: insert clip {Path(ins.media).name} carries no .blurred marker; "
                "blur it first, or bypass with --skip-blur."
            )
        out.append(ins)
    return out


def probe_media(path: Path, *, run=subprocess.run) -> MediaInfo:
    """Size, frame rate and duration of ``path`` via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration", str(path),
    ]  # fmt: skip
    res = run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise InsertError(f"ffprobe failed for {path}: {res.stderr}")
    data = json.loads(res.stdout)
    st = data["streams"][0]
    num, _, den = st["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return MediaInfo(int(st["width"]), int(st["height"]), fps, float(data["format"]["duration"]))


def _load_still_default(path: Path) -> Image.Image:
    from .route_story import hud

    return hud._load_still(path)


def _render_card(still: Image.Image, width: int, height: int, style: str) -> Image.Image:
    from .route_story.hud import still_card_frame

    if style == "full":
        return still_card_frame(still, width, height, 0.0)
    scale = min(width / still.width, height / still.height)
    fitted = still.resize(
        (max(1, round(still.width * scale)), max(1, round(still.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return still_card_frame(canvas, width, height, 0.0)


def plan_inserts(
    lf_path: Path,
    inserts: list[Insert],
    *,
    profile,
    rendition,
    probe: Callable[[Path], MediaInfo] = probe_media,
    keyframes: Callable[[Path], Optional[list[float]]] = probe_keyframes,
    load_still: Callable[[Path], Image.Image] = _load_still_default,
    work_dir: Path,
    gate: Optional[InsertGate] = None,
) -> list[InsertPlan]:
    """Plan every insert; raises before any encode (bad style, refused upscale).

    With a ``gate`` every insert first passes the county, blur and GPS-strip
    gates (:func:`gate_inserts`) and a still is drawn from its gated copy.

    Cut points come from the LF's probed keyframes (``keycut.probe_keyframes``),
    not ``GOP_DURATION_SEC`` arithmetic: encoders add scene-cut keyframes, so a
    fixed grid can name a time that is not a keyframe and break the stream copy.
    """
    for ins in inserts:
        if ins.style not in STYLES:
            raise ValueError(f"insert style must be one of {STYLES}, got {ins.style!r}")
    if gate is not None:
        inserts = gate_inserts(inserts, gate, work_dir=work_dir)
    lf = probe(lf_path)
    kfs = keyframes(lf_path)
    if not kfs:
        raise InsertError(f"no keyframe index for {lf_path}; cannot stream-copy around inserts")
    work_dir.mkdir(parents=True, exist_ok=True)
    target = replace(rendition, width=lf.width, height=lf.height)
    plans: list[InsertPlan] = []
    for i, ins in enumerate(sorted(inserts, key=lambda x: x.at_s)):
        card_path, flt, up = None, "", None
        if Path(ins.media).suffix.lower() in STILL_SUFFIXES:
            kind, dur = "still", float(ins.hold_s)
            card = _render_card(load_still(ins.media), lf.width, lf.height, ins.style)
            card_path = work_dir / f"insert_{i:02d}_card.png"
            card.save(card_path)
        else:
            kind, m = "clip", probe(ins.media)
            dur = m.duration
            up_plan = upscale_plan(m.width, m.height, target, profile.upscale)
            if up_plan is not None:
                flt, up = up_plan.filter, list(up_plan.source)
            else:
                flt = (
                    f"scale={lf.width}:{lf.height}:force_original_aspect_ratio=increase,"
                    f"crop={lf.width}:{lf.height}"
                )
        cut_in, cut_out = keyframe_span(ins.at_s, ins.at_s + dur, kfs, lf.duration)
        if plans and cut_in < plans[-1].cut_out:
            raise InsertError(f"insert at {ins.at_s}s shares a GOP with the previous insert")
        plans.append(
            InsertPlan(ins, kind, cut_in, cut_out, dur, card_path, flt, up, lf.fps, lf.duration)
        )
    return plans


def _encode_argv(lf_path: Path, plan: InsertPlan, out: Path) -> list[str]:
    a = plan.insert.at_s - plan.cut_in
    b = a + plan.duration
    window = f"enable='between(t,{a:.3f},{b:.3f})'"
    argv = ["ffmpeg", "-y", "-ss", f"{plan.cut_in}", "-to", f"{plan.cut_out}", "-i", str(lf_path)]
    if plan.kind == "still":
        argv += ["-loop", "1", "-framerate", f"{plan.lf_fps}", "-t", f"{plan.duration}", "-i"]
        argv += [str(plan.card_path)]
        graph = (
            f"[1:v]setpts=PTS-STARTPTS+{a:.3f}/TB[c];[0:v][c]overlay=eof_action=pass:{window}[v]"
        )
        maps = ["-map", "[v]", "-map", "0:a"]  # the LF's attenuated audio stays under a still
    else:
        argv += ["-i", str(plan.insert.media)]
        graph = (
            f"[1:v]{plan.scaler_filter},setpts=PTS-STARTPTS+{a:.3f}/TB[c];"
            f"[0:v][c]overlay=eof_action=pass:{window}[v];"
            f"[0:a]volume=0:{window}[la];"
            f"[1:a]adelay={round(a * 1000)}:all=1[ca];"
            "[la][ca]amix=inputs=2:normalize=0:duration=first[a]"
        )
        maps = ["-map", "[v]", "-map", "[a]"]  # the clip's own audio, LF dropped for the span
        maps += ["-map_metadata", "0"]  # tags from the LF only, never the clip's location tags
    return [*argv, "-filter_complex", graph, *maps, *lf_reencode_args(), str(out)]


def apply_inserts(
    lf_path: Path, plans: list[InsertPlan], output_path: Path, *, run=subprocess.run
) -> InsertResult:
    """Stream-copy outside each ``[cut_in, cut_out)``, re-encode inside, re-concat."""
    output_path = Path(output_path)
    work = output_path.parent / f"{output_path.stem}_inserts"
    work.mkdir(parents=True, exist_ok=True)
    if not plans:
        raise InsertError("no inserts to apply")
    total = plans[0].lf_duration
    result = InsertResult(output=output_path)
    parts: list[Path] = []

    def step(argv: list[str], part: Path, span: Span) -> None:
        res = run(argv, capture_output=True)
        if res.returncode != 0:
            raise InsertError(f"ffmpeg failed for span {span.start}-{span.end}")
        parts.append(part)
        result.spans.append(span)

    cursor = 0.0
    for i, plan in enumerate(plans):
        if plan.cut_in > cursor:
            part = work / f"part_{i:02d}_head.mp4"
            step(
                copy_range_argv(lf_path, cursor, plan.cut_in, part),
                part,
                Span(cursor, plan.cut_in, True),
            )
        part = work / f"part_{i:02d}_insert.mp4"
        step(_encode_argv(lf_path, plan, part), part, Span(plan.cut_in, plan.cut_out, False))
        if plan.upscaled_from:
            result.upscaled.append(plan.upscaled_from)
        cursor = plan.cut_out
    if cursor < total:
        part = work / "part_tail.mp4"
        step(copy_range_argv(lf_path, cursor, total, part), part, Span(cursor, total, True))
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    res = run(concat_copy_argv(listing, output_path), capture_output=True)
    if res.returncode != 0:
        raise InsertError("ffmpeg concat failed")
    return result


# ── The drive run ────────────────────────────────────────────────────────────
# ``--insert-photos <dir>`` and ``--insert-clip <path>@<mm:ss>`` become Inserts
# per long-form before any encode (placement, the audit sidecar and the
# preflight print all read the same list), and the splice after the OST burn
# ALWAYS carries a real InsertGate: a drive run has no "ungated" mode.

_CLIP_AT_RE = re.compile(r"(\d+):([0-5]\d)")
INSERT_HOLD_S = 4.0


def parse_insert_clip(spec: str) -> tuple[Path, float]:
    """``"PATH@MM:SS"`` -> ``(PATH, seconds)``; the LAST ``@`` splits, so a path may hold one."""
    path, sep, at = str(spec).rpartition("@")
    m = _CLIP_AT_RE.fullmatch(at.strip())
    if not sep or not path.strip() or not m:
        raise InsertError(f"--insert-clip {spec!r}: expected PATH@MM:SS")
    return Path(path), float(int(m[1]) * 60 + int(m[2]))


def photo_files(directory) -> list[Path]:
    """The still files directly inside ``directory``, sorted by name."""
    d = Path(directory)
    if not d.is_dir():
        raise InsertError(f"--insert-photos {directory}: not a directory")
    found = sorted(p for p in d.iterdir() if p.suffix.lower() in STILL_SUFFIXES)
    if not found:
        raise InsertError(f"--insert-photos {directory}: no photos in it")
    return found


@dataclass(frozen=True)
class DriveInserts:
    """The drive's requested inserts. Clips land in the first LF of ``clip_date``."""

    photos: tuple[Path, ...] = ()
    clips: tuple[tuple[Path, float], ...] = ()
    clip_date: str = ""

    @classmethod
    def from_drive(cls, drive, *, clip_date: str = "") -> Optional["DriveInserts"]:
        raw_photos = getattr(drive, "insert_photos", "")
        raw_clips = getattr(drive, "insert_clips", [])
        photos = (
            tuple(photo_files(raw_photos)) if isinstance(raw_photos, str) and raw_photos else ()
        )
        clips = (
            tuple(parse_insert_clip(c) for c in raw_clips) if isinstance(raw_clips, list) else ()
        )
        for path, _ in clips:
            if not path.is_file():
                raise InsertError(f"--insert-clip {path}: no such file")
        return cls(photos, clips, clip_date) if photos or clips else None


def apply_insert_flags(args, config) -> None:
    """Mirror the drive flags onto ``config.drive`` (per invocation, like ``--skip-blur``)."""
    # Only real flag values mirror: argparse gives a str and a list, while a
    # stand-in ``args`` (handler unit tests) answers every attribute with a non-path.
    photos = getattr(args, "insert_photos", None)
    if photos and isinstance(photos, (str, Path)):
        config.drive.insert_photos = str(photos)
    clips = getattr(args, "insert_clip", None)
    if clips and isinstance(clips, (list, tuple)):
        config.drive.insert_clips = list(clips)


def insert_request(drive) -> Optional[dict]:
    """The insert flags as a JSON-safe record (``None`` when none were given).

    Persisted on the state as ``insert_request`` so a resume can tell whether it carries the
    flags the finished days were rendered with.
    """
    photos = getattr(drive, "insert_photos", "")
    clips = getattr(drive, "insert_clips", [])
    photos = photos if isinstance(photos, str) else ""
    clips = sorted(str(c) for c in clips) if isinstance(clips, list) else []
    return {"photos": photos, "clips": clips} if photos or clips else None


def reconcile_insert_request(state, drive, completed_dates) -> None:
    """Refuse a resume whose insert flags differ from those its finished days were rendered with.

    The flags are not restored from state (they stay per-invocation); a finished day is never
    re-rendered, so a changed request could only leave the drive half with, half without the
    inserts. With nothing finished the request is free to change. Records the current request.
    """
    current = insert_request(drive)
    saved = getattr(state, "insert_request", None) or None
    if completed_dates and saved != current:
        raise InsertError(
            f"resume: day(s) {', '.join(sorted(str(d) for d in completed_dates))} were rendered "
            f"with insert flags {saved or 'none'}, this run has {current or 'none'}. "
            "Re-run with the original --insert-photos/--insert-clip flags, or pass --fresh."
        )
    state.insert_request = current


def _photo_county(photo: Path, county_of) -> tuple[str, Optional[tuple[str, str]]]:
    """``(text, (county, state) | None)`` for a photo's EXIF capture position."""
    pos = _photo_gps(photo)
    if pos is None:
        return "no EXIF GPS", None
    try:
        county, state = county_of(*pos)
    except Exception as exc:  # noqa: BLE001 - a dry-run note never breaks the plan
        logger.warning("insert county lookup failed for %s: %s", photo.name, exc)
        county, state = None, None
    if not county:
        return "county unresolved", None
    st = (state or "").strip()
    st = st.upper() if len(st) == 2 else US_STATE_ABBREVS.get(st.title(), st)
    return (f"{county}, {st}" if st else county), (county, state or "")


def describe_drive_inserts(drive) -> list[str]:
    """One preflight line per requested insert; raises on media that is not there."""
    di = DriveInserts.from_drive(drive)
    if di is None:
        return []
    lines = [f"photo  {p.name}  (placed by capture clock against the day's GPX)" for p in di.photos]
    for path, at in di.clips:
        lines.append(
            f"clip   {path.name}  at {int(at) // 60:02d}:{int(at) % 60:02d} of the first long-form"
        )
    return lines


def describe_insert_label_match(drive, lf_labels: dict, county_of) -> list[str]:
    """One line per insert photo: its capture county and the LFs whose label accepts it.

    ``lf_labels`` is ``{lf_index: label}``. Empty when no photos were requested. The county
    test is the insert gate's (``county_in_label``), run against the labels passed in, which
    may differ from the ``per_lf_locations`` the gate sees (overrides, multi-county labels).
    """
    di = DriveInserts.from_drive(drive)
    if di is None or not di.photos:
        return []
    counties = {idx: label_counties(label) for idx, label in lf_labels.items()}
    lines = []
    for p in di.photos:
        text, found = _photo_county(p, county_of)
        if found is None:
            lines.append(f"{p.name}  {text}")
            continue
        ok = [idx for idx, c in sorted(counties.items()) if county_in_label(*found, c)]
        verdict = (
            "accepted by " + ", ".join(f"LF {i}" for i in ok) if ok else "outside every LF label"
        )
        lines.append(f"{p.name}  {text}  ->  {verdict}")
    return lines


def geocoder_county_of(geocoder) -> Callable[[float, float], tuple[Optional[str], Optional[str]]]:
    """Adapt a ``reverse_geocode_detailed`` geocoder to the gate's ``county_of`` shape."""

    def county_of(lat: float, lon: float):
        detail = geocoder.reverse_geocode_detailed(lat, lon) or {}
        return detail.get("county"), detail.get("state")

    return county_of


def master_to_lf_seconds(
    master_s: float, ranges: Sequence[tuple[float, float]], lead_s: float
) -> Optional[float]:
    """Day-master second -> second in the LF built from ``ranges`` behind a ``lead_s`` montage.

    Approximate to about one GOP: the extractor merges sub-GOP gaps and snaps cuts
    to keyframes. ``None`` when the LF does not contain that second.
    """
    acc = lead_s
    for start, end in ranges:
        if start <= master_s < end:
            return acc + (master_s - start)
        acc += end - start
    return None


def lf_to_master_seconds(
    lf_s: float, ranges: Sequence[tuple[float, float]], lead_s: float
) -> Optional[float]:
    """Inverse of :func:`master_to_lf_seconds`; the montage lead maps to the first range's start."""
    acc = lead_s
    for start, end in ranges:
        if lf_s < acc + (end - start):
            return start + max(0.0, lf_s - acc)
        acc += end - start
    return None


MAX_TRACK_GAP_S = 60.0


def _track_position(points, master_s: Optional[float]) -> Optional[tuple[float, float]]:
    """The day-track fix nearest ``master_s``; ``None`` past ``MAX_TRACK_GAP_S`` (fail closed)."""
    if master_s is None or not points:
        return None
    t, lat, lon = min(points, key=lambda p: abs(float(p[0]) - master_s))
    return (float(lat), float(lon)) if abs(float(t) - master_s) <= MAX_TRACK_GAP_S else None


def _photo_capture(photo: Path) -> Optional[datetime]:
    from .photos.exif_gps import extract_exif_capture_timestamp

    return extract_exif_capture_timestamp(photo)


def _photo_gps(photo: Path) -> Optional[tuple[float, float]]:
    from .photos.exif_gps import extract_exif_gps

    return extract_exif_gps(photo)


def _master_seconds_of(capture: datetime, episodes, boundaries) -> Optional[float]:
    """Master second of a camera-clock capture time; ``None`` outside every episode.

    Photos and episodes share the camera's naive wall clock, so the match is on
    that clock and never on a guessed zone.
    """
    capture = capture.replace(tzinfo=None)
    for ep, (start, end) in zip(episodes, boundaries):
        began = getattr(ep, "creation_time", None)
        if began is None:
            continue
        off = (capture - began.replace(tzinfo=None)).total_seconds()
        if 0.0 <= off < float(end) - float(start):
            return float(start) + off
    return None


def day_photo_master_seconds(
    photos: Sequence[Path], episodes, boundaries, *, capture_of=_photo_capture
) -> dict[Path, Optional[float]]:
    """Each photo's master second on this day (``None`` = not on this day)."""
    out: dict[Path, Optional[float]] = {}
    for photo in photos:
        capture = capture_of(photo)
        if capture is None:
            raise InsertError(f"{Path(photo).name}: no capture time, cannot place it")
        out[photo] = _master_seconds_of(capture, episodes, boundaries)
    return out


def place_day_inserts(
    *,
    di: DriveInserts,
    date: str,
    episodes,
    boundaries,
    gps_points,
    lf_ranges: dict[int, list[tuple[float, float]]],
    lead_s: float,
    capture_of=_photo_capture,
    gps_of=_photo_gps,
) -> dict[int, list[Insert]]:
    """``{lf_index: [Insert, ...]}`` for one day, at LF time; raises on anything unplaceable.

    A photo on this day is anchored on the day's track (``anchor_photo_insert``),
    then moved onto the LF that holds that second; one no LF holds, or one the
    track cannot place, is refused rather than guessed. A photo on no episode of
    this day belongs to another day and is left alone (the run checks that every
    photo lands on some day before it encodes). Clips go to the first LF of
    ``di.clip_date`` at their ``@mm:ss``.
    """
    placed: dict[int, list[Insert]] = {}
    on_day = {
        p: s
        for p, s in day_photo_master_seconds(
            di.photos, episodes, boundaries, capture_of=capture_of
        ).items()
        if s is not None
    }
    if on_day:
        started = episodes[0].creation_time.replace(tzinfo=timezone.utc)
        track = [[RoutePoint(float(t), float(la), float(lo)) for t, la, lo in (gps_points or [])]]
        total = float(boundaries[-1][1])
        for photo, master_s in on_day.items():
            anchored = anchor_photo_insert(
                photo,
                captured_at_utc=started + timedelta(seconds=master_s),
                track_started_at_utc=started,
                segments=track,
                lf_start_s=0.0,
                lf_duration_s=total,
                gps=gps_of(photo),
                hold_s=INSERT_HOLD_S,
            )
            for idx in sorted(lf_ranges):
                lf_s = master_to_lf_seconds(anchored.at_s, lf_ranges[idx], lead_s)
                if lf_s is not None:
                    placed.setdefault(idx, []).append(replace(anchored, at_s=lf_s))
                    break
            else:
                raise InsertError(
                    f"{Path(photo).name}: captured at master {anchored.at_s:.0f}s of {date}, "
                    "inside no long-form of the day"
                )
    if di.clips and date == di.clip_date and lf_ranges:
        first = min(lf_ranges)
        length = lead_s + sum(e - s for s, e in lf_ranges[first])
        for path, at in di.clips:
            if at >= length:
                raise InsertError(
                    f"{path.name}: @{int(at) // 60:02d}:{int(at) % 60:02d} is past the end of "
                    f"long-form {first} ({length:.0f}s)"
                )
            # A clip has no capture position of its own: it is where the track was at
            # the cut point it was asked to sit at, so the county gate can run on it.
            spot = _track_position(gps_points, lf_to_master_seconds(at, lf_ranges[first], lead_s))
            placed.setdefault(first, []).append(Insert(at, path, style="full", capture=spot))
    return {i: sorted(v, key=lambda x: x.at_s) for i, v in sorted(placed.items())}


def insert_records(placed: dict[int, list[Insert]]) -> dict[int, list[dict]]:
    """JSON-safe audit records per LF: what each insert is and where it cuts in."""
    out = {}
    for idx, items in placed.items():
        out[idx] = [
            {
                "kind": "still" if Path(i.media).suffix.lower() in STILL_SUFFIXES else "clip",
                "media": Path(i.media).name,
                "at_s": round(float(i.at_s), 3),
                "style": i.style,
                **(
                    {"hold_s": float(i.hold_s)}
                    if Path(i.media).suffix.lower() in STILL_SUFFIXES
                    else {}
                ),
            }
            for i in items
        ]
    return out


def shift_photo_inserts(inserts: Sequence[Insert], delta_s: float) -> list[Insert]:
    """Move the footage-anchored stills by ``delta_s`` (the montage's real minus planned length).

    A clip's ``@mm:ss`` is a time on the finished LF, so it stays where it was asked.
    """
    return [
        replace(i, at_s=max(0.0, i.at_s + delta_s))
        if Path(i.media).suffix.lower() in STILL_SUFFIXES
        else i
        for i in inserts
    ]


def splice_lf_inserts(
    lf_path: Path,
    inserts: Sequence[Insert],
    *,
    config,
    label: Optional[str],
    profile,
    work_dir: Path,
    plan: Callable[..., list[InsertPlan]] = plan_inserts,
    apply: Callable[..., InsertResult] = apply_inserts,
) -> InsertResult:
    """Splice ``inserts`` into the finished LF in place, behind an always-present gate.

    The gate is built here, never taken from the caller: with no ``config`` it is
    the closed one (a clip is unproven-blurred, a still goes through the privacy
    gate under the run's default), so nothing reaches the encode ungated.
    """
    lf_path = Path(lf_path)
    gate = (
        InsertGate.from_config(config, label)
        if config is not None
        else InsertGate(label_counties(label))
    )
    plans = plan(
        lf_path,
        list(inserts),
        profile=profile,
        rendition=RENDITIONS["youtube_longform"],
        work_dir=Path(work_dir),
        gate=gate,
    )
    out = lf_path.with_name(f"{lf_path.stem}.inserted{lf_path.suffix}")
    result = apply(lf_path, plans, out)
    os.replace(out, lf_path)
    shutil.rmtree(out.parent / f"{out.stem}_inserts", ignore_errors=True)
    return result


def splice_day_lf_inserts(
    lf_path: Path,
    inserts: Sequence[Insert],
    *,
    config,
    label: Optional[str],
    profile,
    work_dir: Path,
    montage_planned_s: float,
    prepended: Path,
    concatenated: Path,
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> InsertResult:
    """Splice a day LF's inserts after its OST burn, moving stills by the montage's real lead.

    The lead is the probed length the montage added (``prepended`` minus
    ``concatenated``); stills were placed against ``montage_planned_s``.
    """
    lead = float(probe(Path(prepended)).duration) - float(probe(Path(concatenated)).duration)
    shifted = shift_photo_inserts(inserts, lead - float(montage_planned_s))
    return splice_lf_inserts(
        lf_path, shifted, config=config, label=label, profile=profile, work_dir=work_dir
    )


def plan_day_from_batch(
    di: DriveInserts,
    day_batch,
    lf_plans,
    gps_points,
    *,
    lead_s: float,
    capture_of=_photo_capture,
    gps_of=_photo_gps,
) -> dict[int, list[Insert]]:
    """``place_day_inserts`` for one ``DayBatch``: derives the boundaries and each LF's master ranges."""
    from keycut import boundaries_from_durations, segments_to_master_ranges

    from .montage_builder import to_cut_segment

    episodes = list(getattr(day_batch, "episodes", None) or [])
    boundaries = boundaries_from_durations(
        float(getattr(e, "total_duration", 0.0) or 0.0) for e in episodes
    )
    lf_ranges: dict[int, list[tuple[float, float]]] = {}
    for i, plan in enumerate(lf_plans or []):
        segs = list(getattr(plan, "segments", []) or [])
        if segs:
            idx = int(getattr(plan, "index", i + 1))
            lf_ranges[idx] = [
                (float(s), float(e))
                for s, e in segments_to_master_ranges((to_cut_segment(s) for s in segs), boundaries)
            ]
    return place_day_inserts(
        di=di,
        date=day_batch.date,
        episodes=episodes,
        boundaries=boundaries,
        gps_points=gps_points,
        lf_ranges=lf_ranges,
        lead_s=lead_s,
        capture_of=capture_of,
        gps_of=gps_of,
    )


def photos_unplaced(di: DriveInserts, day_batches, *, capture_of=_photo_capture) -> list[str]:
    """Names of requested photos that fall on no episode of any day (checked before any encode)."""
    from keycut import boundaries_from_durations

    left = {p: False for p in di.photos}
    for batch in day_batches:
        eps = list(getattr(batch, "episodes", None) or [])
        bounds = boundaries_from_durations(
            float(getattr(e, "total_duration", 0.0) or 0.0) for e in eps
        )
        for p, s in day_photo_master_seconds(di.photos, eps, bounds, capture_of=capture_of).items():
            if s is not None:
                left[p] = True
    return [Path(p).name for p, hit in left.items() if not hit]


def insert_label(
    per_lf_locations: Optional[dict], idx: int, fallback: Optional[str]
) -> Optional[str]:
    """The county label the gate checks an LF against (``per_lf_locations``, not the OST text)."""
    return (per_lf_locations or {}).get(idx) or fallback


def plan_drive_inserts(
    config,
    day_batches,
    *,
    lf_plans_by_date: Optional[dict],
    gps_points_by_date: Optional[dict],
    label_of: Callable[[str, int], Optional[str]],
    place: Callable[..., dict[int, list[Insert]]] = plan_day_from_batch,
    gate_kw: Optional[dict] = None,
    render_dates: Optional[Collection[str]] = None,
) -> dict[str, dict[int, list[Insert]]]:
    """Place every requested insert on the whole drive, gate the days being rendered, before any encode.

    ``day_batches`` is EVERY day of the drive, finished or not (a resume must still place a
    photo of a finished day, and ``clip_date`` is the drive's first day, not the first one left
    to render). ``render_dates`` names the days this run renders; only their inserts are gated
    (county, blur, GPS-strip), because a finished day already carries its inserts. ``None``
    gates every day, as a fresh run does.

    ``{date: {lf_index: [Insert, ...]}}``. Raises ``InsertError``/``BlurGateError`` on the
    first insert that is unplaceable or refused, so a refusal costs no OST burn or master.
    The splice gates again (it must stay safe alone); stills' gated copies made here are dropped.
    """
    di = DriveInserts.from_drive(
        getattr(config, "drive", None), clip_date=str(min(b.date for b in day_batches))
    )
    if di is None:
        return {}
    lost = photos_unplaced(di, day_batches) if di.photos else []
    if lost:
        raise InsertError(f"--insert-photos: no day of this drive holds {', '.join(lost)}")
    lead = float(getattr(config.drive, "montage_max_duration_sec", 15.0))
    out: dict[str, dict[int, list[Insert]]] = {}
    with tempfile.TemporaryDirectory(prefix="insert_gate_") as tmp:
        for batch in day_batches:
            placed = place(
                di,
                batch,
                (lf_plans_by_date or {}).get(batch.date) or [],
                (gps_points_by_date or {}).get(batch.date),
                lead_s=lead,
            )
            if render_dates is None or batch.date in render_dates:
                for idx, items in placed.items():
                    gate = InsertGate.from_config(
                        config, label_of(batch.date, idx), **(gate_kw or {})
                    )
                    gate_inserts(items, gate, work_dir=Path(tmp) / f"{batch.date}_{idx}")
            out[batch.date] = placed
    return out
