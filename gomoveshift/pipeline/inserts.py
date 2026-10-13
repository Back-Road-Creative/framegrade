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
import re
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from keycut import probe_keyframes
from PIL import Image

from .blur_gate import BlurGateError, blur_required, master_blur_proven
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
    key = re.sub(r" county$", "", county.strip().lower())
    st = (state or "").strip()
    st = st.upper() if len(st) == 2 else US_STATE_ABBREVS.get(st.title(), "")
    # Fail closed on state: a county name is only unique within its state. A label
    # pair with a state needs an insert whose state is known and equal; a state-less
    # pair matches on the name alone only when it is the label's sole state-less pair.
    lone_stateless = sum(1 for _, s in gate.label_counties if not s) == 1
    if not any(key == c and (st == s if s else lone_stateless) for c, s in gate.label_counties):
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
