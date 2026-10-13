"""Insert engine: key-cut stills and clips into a rendered long-form.

``plan_inserts`` decides where each insert cuts in and out; ``apply_inserts``
stream-copies everything outside the bounding GOPs and re-encodes only
``[cut_in, cut_out)``. See ``docs/pipeline/153-insert-engine.md``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from keycut import probe_keyframes
from PIL import Image

from .montage_builder import (
    GOP_DURATION_SEC,  # noqa: F401  (nominal GOP; the cut uses probed keyframes)
    concat_copy_argv,
    copy_range_argv,
    keyframe_span,
    lf_reencode_args,
)
from .quality import upscale_plan

logger = logging.getLogger(__name__)

STYLES = ("full", "letterbox")
STILL_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic"}


class InsertError(RuntimeError):
    """An insert cannot be planned or applied."""


@dataclass(frozen=True)
class Insert:
    """One insert: ``media`` shown from ``at_s``. A clip ignores ``hold_s``."""

    at_s: float
    media: Path
    hold_s: float = 4.0
    style: str = "full"


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
) -> list[InsertPlan]:
    """Plan every insert; raises before any encode (bad style, refused upscale).

    Cut points come from the LF's probed keyframes (``keycut.probe_keyframes``),
    not ``GOP_DURATION_SEC`` arithmetic: encoders add scene-cut keyframes, so a
    fixed grid can name a time that is not a keyframe and break the stream copy.
    """
    for ins in inserts:
        if ins.style not in STYLES:
            raise ValueError(f"insert style must be one of {STYLES}, got {ins.style!r}")
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
