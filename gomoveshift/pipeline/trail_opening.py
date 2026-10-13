"""The trail film as the LF's opening.

When ``--trail`` ran, the film's satellite opening + first map leg (capped at
``config.drive.montage_max_duration_sec``) replaces the peak montage at 0 s of the LF the film
was placed from. The film exists only after that LF rendered, so the swap is a post-film step on
the FINAL LF, and the day's intermediates are already purged by then:

* the LF's lead (the montage length, probed at render time) and its OST recipe are recorded in
  ``lf_NN.lead.json`` while the intermediates still exist;
* the opening is cut from the film, silenced, and burned with the SAME audited OST text (the
  closing website CTA stays off: it belongs to the LF's tail);
* the body is the final LF stream-copied from the first keyframe at or after the lead, with the
  few frames before it re-encoded; every insert already spliced in rides with the body, so it
  still lands on its footage (a clip pinned to an ``@mm:ss`` would not, so such an LF keeps its
  montage);
* a ``lf_NN.trail_opening.json`` marker (``pending`` before the atomic replace, ``done`` after)
  makes a crash resumable, and its key is folded into the LF's freshness fingerprint, so a
  resume neither re-burns a swapped LF nor keeps a swap whose LF was re-burned.

A film whose opening cannot be located, or an LF that cannot take the swap, is a recorded skip
(``Deliverables.trail_skips``), never a silent fallback: the LF keeps its montage (or, on a
re-rendered film, the earlier opening already swapped in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from keycut import probe_keyframes

from ..console.settings import current_brand
from ..publishing.brand_policy import load_media_profile
from .inserts import STILL_SUFFIXES, InsertError, MediaInfo, probe_media
from .montage_builder import (
    concat_copy_argv,
    copy_range_argv,
    keyframe_span,
    reencode_range_argv,
)

if TYPE_CHECKING:  # media_api imports day_pipeline, which imports this module
    from .media_api import Deliverables, TrailFilm

logger = logging.getLogger(__name__)

#: An opening shorter than this is a flash, not an opening.
MIN_OPENING_S = 1.5
#: How far the swapped file's length may stray from the planned one.
DURATION_TOLERANCE_S = 0.5


class OpeningSkipError(Exception):
    """The reason an LF keeps its montage."""


def _lead_path(lf_dir: Path, idx: int) -> Path:
    return lf_dir / f"lf_{idx:02d}.lead.json"


def _marker_path(lf_dir: Path, idx: int) -> Path:
    return lf_dir / f"lf_{idx:02d}.trail_opening.json"


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    os.replace(tmp, path)


def record_lead(
    lf_dir: Path,
    idx: int,
    *,
    lead_s: float,
    location: str,
    crossfade_sec: float,
    brand_logo_path: Optional[str],
    inserts: Sequence[Any] = (),
    planned_s: Optional[float] = None,
) -> None:
    """Save what a later swap needs from a burn: the montage's real length and the OST recipe."""
    clips = [i for i in inserts if Path(str(i.media)).suffix.lower() not in STILL_SUFFIXES]
    _write_json(
        _lead_path(lf_dir, idx),
        {
            "lead_s": float(lead_s),
            "planned_s": float(lead_s if planned_s is None else planned_s),
            "location": location,
            "crossfade_sec": float(crossfade_sec),
            "brand_logo_path": brand_logo_path,
            "has_clip_inserts": bool(clips),
            "min_insert_s": min((float(i.at_s) for i in inserts), default=None),
        },
    )


def fold(base_fingerprint: str, key: str) -> str:
    """The LF fingerprint of a swapped LF: the burn's own, with the opening folded in."""
    return hashlib.sha256(f"{base_fingerprint}\x00trail-opening\x00{key}".encode()).hexdigest()


def expected_fingerprint(lf_dir: Path, idx: int, base_fingerprint: str) -> str:
    """What ``lf_NN.ost.sha256`` holds for a fresh LF: folded once the swap is done."""
    marker = _read_json(_marker_path(lf_dir, idx))
    if marker and marker.get("status") == "done" and marker.get("key"):
        return fold(base_fingerprint, str(marker["key"]))
    return base_fingerprint


def reset(lf_dir: Path, idx: int) -> None:
    """Forget a swap: the LF is about to be re-burned, so it has its montage again."""
    for path in (_marker_path(lf_dir, idx), _lead_path(lf_dir, idx)):
        path.unlink(missing_ok=True)


def _opening_seconds(receipt_path: Optional[Path], cap_s: float) -> tuple[float, str]:
    """``(seconds, key)`` of the film's opening, from its receipt; a skip if it can't be found."""
    receipt = _read_json(Path(receipt_path)) if receipt_path else None
    block = (receipt or {}).get("opening")
    if not isinstance(block, dict):
        raise OpeningSkipError("the film receipt records no opening span")
    fps, end = block.get("fps"), block.get("end_frame")
    if not isinstance(fps, (int, float)) or not isinstance(end, int) or fps <= 0 or end <= 0:
        raise OpeningSkipError("the film receipt's opening span is unusable")
    seconds = min(end / float(fps), float(cap_s))
    if seconds < MIN_OPENING_S:
        raise OpeningSkipError(f"the film's opening is {seconds:.2f}s, under {MIN_OPENING_S}s")
    return seconds, hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest()


def _ffmpeg(run, argv: list[str], what: str) -> None:
    res = run(argv, capture_output=True)
    if res.returncode != 0:
        raise OpeningSkipError(f"ffmpeg failed to {what}")


def _has_audio(run, path: Path) -> bool:
    res = run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )  # fmt: skip
    return res.returncode == 0 and bool(str(res.stdout).strip())


def _video_duration(run, path: Path) -> Optional[float]:
    """The first video stream's own duration, or ``None`` when ffprobe cannot give one.

    The format duration follows the longest stream, so a join whose video timestamps collapsed
    still reports the audio's length; only the stream shows it.
    """
    res = run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if res.returncode != 0:
        return None
    try:
        return float(str(res.stdout).strip().splitlines()[0].rstrip(","))
    except (ValueError, IndexError):
        return None


def _exact_fps(fps: float) -> str:
    """The LF's frame rate as the rational ffprobe reported it (30000/1001, not 29.97003).

    A rounded rate gives the opening a different timebase than the stream-copied body, and the
    concat join then collapses the body's timestamps.
    """
    frac = Fraction(float(fps)).limit_denominator(1001)
    return f"{frac.numerator}/{frac.denominator}"


def _cut_opening(run, film: Path, seconds: float, lf: MediaInfo, audio: bool, out: Path) -> None:
    """The film's first ``seconds`` at the LF's size and rate, silent (the near-silent rule)."""
    argv = ["ffmpeg", "-y", "-t", f"{seconds}", "-i", str(film)]
    if audio:
        argv += ["-f", "lavfi", "-t", f"{seconds}", "-i", "anullsrc=r=48000:cl=stereo"]
    argv += ["-map", "0:v:0"]
    if audio:
        argv += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "256k"]
    else:
        argv += ["-an"]
    argv += [
        "-vf", f"scale={lf.width}:{lf.height},fps={_exact_fps(lf.fps)},setsar=1",
        "-c:v", "libx265", "-crf", "18", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
        str(out),
    ]  # fmt: skip
    _ffmpeg(run, argv, "cut the film's opening")


def apply_trail_opening(
    d: Deliverables,
    film: TrailFilm,
    config: Any,
    *,
    media_profile: Any = None,
    run: Callable = subprocess.run,
    probe: Callable[[Path], MediaInfo] = probe_media,
    keyframes: Callable[[Path], Optional[list[float]]] = probe_keyframes,
    burn: Optional[Callable] = None,
) -> Optional[dict]:
    """Swap the LF's montage lead for the film's opening; ``None`` when already swapped.

    Raises :class:`OpeningSkipError` (the LF is left untouched) when it cannot be done.
    """
    pairs = dict(zip(d.regular_lf_indices, d.regular_lf_paths))
    if film.lf_index not in pairs:
        raise OpeningSkipError(f"LF {film.lf_index} is not among the day's LFs")
    lf = Path(pairs[film.lf_index])
    lf_dir, idx = lf.parent, film.lf_index
    fp_path = lf_dir / f"lf_{idx:02d}.ost.sha256"
    cap = float(getattr(config.drive, "montage_max_duration_sec", 15.0))
    opening_s, key = _opening_seconds(film.receipt, cap)

    marker_path = _marker_path(lf_dir, idx)
    marker = _read_json(marker_path) or {}
    if marker.get("status") == "done" and marker.get("key") == key:
        return None
    recipe = _read_json(_lead_path(lf_dir, idx))
    if recipe is None:
        raise OpeningSkipError(
            "no montage lead was recorded for this LF (rendered before the trail opening existed)"
        )
    if recipe.get("has_clip_inserts"):
        raise OpeningSkipError("the LF has an --insert-clip pinned to a time; its lead cannot move")
    lead = float(recipe["lead_s"])  # the inserts sit where the montage put them
    first_insert = recipe.get("min_insert_s")
    if first_insert is not None and float(first_insert) < lead + 0.01:
        raise OpeningSkipError("an insert sits inside the montage lead")
    if not lf.is_file():
        raise OpeningSkipError(f"{lf.name} is missing")

    try:
        info = probe(lf)
    except (InsertError, OSError, ValueError, KeyError) as exc:
        raise OpeningSkipError(f"cannot probe {lf.name}: {exc}") from exc
    # A re-rendered film on a swapped LF replaces the earlier opening in place: a finished day
    # is never re-burned, so the cut is that opening's length, not the montage's.
    lead = float(marker.get("from_s") or marker.get("opening_s") or lead)
    expected = opening_s + (info.duration - lead)
    base_fp = marker.get("base_fp") or (fp_path.read_text().strip() if fp_path.is_file() else "")
    landed = (
        marker.get("status") == "pending"
        and abs(info.duration - marker["new_duration"]) <= 0.25
        and lf.stat().st_size == marker.get("new_size")  # length alone can match the old LF
    )
    if landed:
        pass  # the replace landed and the process died before the marker: finish, don't redo
    else:
        _swap(run, probe, keyframes, burn, lf, Path(film.path), info, lead, opening_s, recipe,
              media_profile, marker_path, key, base_fp, expected)  # fmt: skip
    fp_path.write_text(fold(base_fp, key))
    _write_json(
        marker_path, {"status": "done", "key": key, "base_fp": base_fp, "opening_s": opening_s}
    )
    # Phase D's recorded insert times are planned against the planned lead; the swap moves them.
    planned = float(recipe.get("planned_s", lead))
    return {
        "lf_index": idx,
        "opening_s": round(opening_s, 3),
        "lead_s": round(lead, 3),
        "insert_delta_s": round(opening_s - planned, 3),
    }


def _swap(
    run, probe, keyframes, burn, lf: Path, film: Path, info: MediaInfo, lead: float,
    opening_s: float, recipe: dict, media_profile, marker_path: Path, key: str, base_fp: str,
    expected: float,
) -> None:  # fmt: skip
    if burn is None:
        from .ost_applier import apply_ost as burn
    if media_profile is None:  # the phases pass none; the LF was burned with the brand's
        media_profile = load_media_profile(current_brand())
    kfs = keyframes(lf)
    if not kfs:
        raise OpeningSkipError(f"no keyframe index for {lf.name}")
    _, cut_out = keyframe_span(lead, lead, kfs, info.duration)
    if cut_out >= info.duration:
        raise OpeningSkipError("the LF has no keyframe after its lead")
    work = lf.parent / f".lf_{lf.stem[3:]}_opening"
    work.mkdir(exist_ok=True)
    raw, burned = work / "opening_raw.mp4", work / "opening.mp4"
    head, body, out = work / "head.mp4", work / "body.mp4", work / "swapped.mp4"
    parts = [burned]
    try:
        _cut_opening(run, film, opening_s, info, _has_audio(run, lf), raw)
        done = burn(
            input_path=raw,
            output_path=burned,
            location=recipe["location"],
            layout="regular_lf",
            brand_logo_path=recipe.get("brand_logo_path"),
            prefer_nvenc=True,
            montage_duration=opening_s,
            crossfade_sec=float(recipe["crossfade_sec"]),
            media_profile=media_profile,
            website_cta=False,
        )
        if done is None:
            raise OpeningSkipError("the OST burn of the opening failed")
        if cut_out - lead > 0.05:
            _ffmpeg(run, reencode_range_argv(lf, lead, cut_out, head), "re-encode the lead seam")
            parts.append(head)
        _ffmpeg(run, copy_range_argv(lf, cut_out, info.duration, body), "copy the LF body")
        parts.append(body)
        listing = work / "concat.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
        _ffmpeg(run, concat_copy_argv(listing, out), "join the opening to the LF")
        got = probe(out).duration
        if abs(got - expected) > DURATION_TOLERANCE_S:
            raise OpeningSkipError(f"swapped LF is {got:.2f}s, expected {expected:.2f}s")
        video = _video_duration(run, out)
        if video is None or abs(video - expected) > DURATION_TOLERANCE_S:
            seen = "unreadable" if video is None else f"{video:.2f}s"
            raise OpeningSkipError(
                f"swapped LF's video stream is {seen}, expected {expected:.2f}s "
                "(its timestamps collapsed in the join)"
            )
        _write_json(
            marker_path,
            {
                "status": "pending",
                "key": key,
                "base_fp": base_fp,
                "new_duration": got,
                "new_size": out.stat().st_size,
                "from_s": lead,
            },
        )
        os.replace(out, lf)
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def _kept_lead(d: Deliverables, film: TrailFilm) -> str:
    """What a skipped swap leaves at 0 s: an earlier film opening, else the montage."""
    pairs = dict(zip(d.regular_lf_indices, d.regular_lf_paths))
    lf = pairs.get(film.lf_index)
    marker = _read_json(_marker_path(Path(lf).parent, film.lf_index)) if lf else None
    return "its earlier trail opening" if (marker or {}).get("status") == "done" else "its montage"


def attach_trail_opening(d: Deliverables, film: TrailFilm, config: Any, **kw: Any) -> bool:
    """Swap, recording any skip (once) on ``d``; a swap failure never fails the drive."""
    try:
        record = apply_trail_opening(d, film, config, **kw)
    except OpeningSkipError as skip:
        reason = f"trail opening skipped for {d.date}: {skip}; the LF keeps {_kept_lead(d, film)}"
    except Exception as exc:  # noqa: BLE001 -- a sibling deliverable never fails the drive
        logger.error("  trail: opening swap failed", exc_info=True)
        reason = (
            f"trail opening failed for {d.date}: {type(exc).__name__}: {exc}; "
            f"the LF keeps {_kept_lead(d, film)}"
        )
    else:
        if record is not None:
            logger.info("  trail: LF %d opens on the trail film (%.1fs)", record["lf_index"],
                        record["opening_s"])  # fmt: skip
            d.trail_openings = [o for o in d.trail_openings if o["lf_index"] != record["lf_index"]]
            d.trail_openings.append(record)
        return record is not None
    logger.warning("  trail: %s", reason)
    if reason not in d.trail_skips:
        d.trail_skips.append(reason)
    return False
