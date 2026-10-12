"""Wires ``select.py``'s best-of-look-alike-group picker into the film CLI.

Two concerns live here, both pure glue around ``select.py`` (which stays
untouched -- it only knows about decoded previews, never files or CLI flags):

* Turning ``PhotoMoment``\\ s into ``PhotoCandidate``\\ s (decode + EXIF route
  time + resolved placement) and running the selector.
* Keeping stills from crowding out the map: a pacing cap that shrinks
  ``--photo-seconds`` when the kept stills would otherwise eat too much of
  the film's runtime.
"""

from __future__ import annotations

import functools
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image

from .assembly import MAP_SETTLE_SECONDS
from .hud import _load_still
from .select import PhotoCandidate, SelectionOptions, SelectionResult, select_best_photos

if TYPE_CHECKING:
    from .film import PhotoMoment

logger = logging.getLogger(__name__)

# Photos graded at once. Each grade holds ~2.2 GB at its peak (a 20.7 MP
# camera still, measured) and runs ~55 s on one core, so a handful
# in parallel turns a render's serial quarter hour into a few minutes without
# starving the box. ``GMS_ROUTE_STORY_STILL_WORKERS`` overrides; 1 is serial.
_DEFAULT_STILL_WORKERS = 4


def _still_workers() -> int:
    try:
        return max(1, int(os.environ.get("GMS_ROUTE_STORY_STILL_WORKERS", "")))
    except ValueError:
        return _DEFAULT_STILL_WORKERS


def load_stills(paths: Sequence[Path], *, still_source: str) -> list[Image.Image]:
    """Every photo's graded still, in ``paths`` order, graded in parallel.

    Each worker also writes the grade cache (``graded_cache``), so the beat
    that later shows a kept photo reads it back instead of grading it again.
    """
    load = functools.partial(_load_still, source=still_source)
    workers = min(_still_workers(), len(paths))
    if workers <= 1:
        return [load(path) for path in paths]
    # Spawned, never forked: by the time stills load, the film may already hold
    # capture threads, and a forked child inherits their locks mid-flight.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(load, paths))


# Stills may not crowd out the map: capped at this share of total film
# runtime, computed from intro + map + kept-stills + recap seconds.
STILLS_SHARE_CAP = 0.35
# A still is never shrunk shorter than this, even when the cap is still
# exceeded at this floor -- a warning is logged instead (``floor_hit``).
PHOTO_SECONDS_FLOOR = 2.5


def build_photo_candidates(
    mapped_photos: Sequence["PhotoMoment"], *, still_source: str
) -> list[PhotoCandidate]:
    """Decode each mapped photo's preview and its EXIF route-clock evidence.

    ``captured_at`` is ``PhotoMoment.captured_at_exif_s`` -- the EXIF shutter
    time on the route clock -- never ``anchor.track_time_s`` (the nearest GPX
    TRACK POINT's time, which can differ from when the shutter fired) and
    never file mtime. A photo with no EXIF capture time raises rather than
    silently grouping it as if it were shot at the route's start; main()
    already refuses such a photo before it is ever mapped, so this should
    never fire in practice.  ``placement`` is the anchor's resolved lat/lon,
    when both are known.
    """
    for photo in mapped_photos:
        if photo.captured_at_exif_s is None:
            raise ValueError(f"{photo.source.name} has no EXIF capture time for selection")
    images = load_stills([photo.source for photo in mapped_photos], still_source=still_source)
    candidates = []
    for photo, image in zip(mapped_photos, images):
        anchor = photo.anchor
        placement = (
            (anchor.latitude, anchor.longitude)
            if anchor.latitude is not None and anchor.longitude is not None
            else None
        )
        candidates.append(
            PhotoCandidate(
                name=photo.source.name,
                image=image,
                captured_at=photo.captured_at_exif_s,
                placement=placement,
            )
        )
    return candidates


def select_kept_photos(
    mapped_photos: Sequence["PhotoMoment"],
    *,
    still_source: str,
    options: SelectionOptions,
) -> tuple[list["PhotoMoment"], SelectionResult]:
    """Run best-of-look-alike selection over already-placed photos.

    ``mapped_photos`` may also include bookend photos (same-day, shot just
    outside the recording) -- ``build_photo_candidates`` decodes any photo
    with an EXIF capture time, and a bookend's ``anchor.latitude``/
    ``longitude`` are already ``None``, so it becomes a candidate with
    ``placement=None`` like any other unplaced-by-position candidate.
    Returns the kept ``PhotoMoment``\\ s, in the same order as ``mapped_photos``
    (not the selector's capture-time order), plus the full ``SelectionResult``
    for the receipt. A truly unplaced photo never reaches here -- the caller
    filters those out before calling this.
    """
    if not mapped_photos:
        result = select_best_photos([], options)
        _log_selection(candidate_count=0, options=options, result=result)
        return [], result
    candidates = build_photo_candidates(mapped_photos, still_source=still_source)
    result = select_best_photos(candidates, options)
    _log_selection(candidate_count=len(candidates), options=options, result=result)
    kept_names = set(result.kept)
    kept = [photo for photo in mapped_photos if photo.source.name in kept_names]
    return kept, result


def _log_selection(
    *, candidate_count: int, options: SelectionOptions, result: SelectionResult
) -> None:
    """One totals line always; one line per group that actually collapsed
    (more than one member) -- a shoot with no look-alikes must not spam a
    line per photo. Selection off logs only that it's off."""
    if not options.enabled:
        logger.info("photo selection disabled; kept all %d candidate(s)", candidate_count)
        return
    dropped = candidate_count - len(result.kept)
    logger.info(
        "photo selection: %d candidate(s) -> %d group(s), kept %d, dropped %d",
        candidate_count,
        len(result.groups),
        len(result.kept),
        dropped,
    )
    for group in result.groups:
        if len(group.members) > 1:
            logger.info(
                "photo selection group: winner=%s losers=%s (%s)",
                group.winner,
                list(group.losers),
                group.reason,
            )


def resolve_photo_seconds(
    *,
    requested_seconds: float,
    kept_count: int,
    intro_seconds: float,
    map_seconds: float,
    recap_seconds: float,
) -> dict[str, object]:
    """Shrink per-still seconds so kept stills stay under ``STILLS_SHARE_CAP``.

    Pure function: given the seconds every other scene will already take
    (``intro_seconds + map_seconds + recap_seconds``) and how many stills
    survived selection, returns the seconds to actually use per still, never
    shrunk below ``PHOTO_SECONDS_FLOOR`` and never enlarged above what was
    requested. ``floor_hit`` is true only when the floor itself still leaves
    stills over the cap.
    """
    other_seconds = intro_seconds + map_seconds + recap_seconds
    used = requested_seconds
    floor_hit = False
    if kept_count > 0 and other_seconds > 0:
        denominator = kept_count * (1 - STILLS_SHARE_CAP)
        if denominator > 0:
            max_seconds = (STILLS_SHARE_CAP * other_seconds) / denominator
            if requested_seconds > max_seconds:
                if max_seconds < PHOTO_SECONDS_FLOOR:
                    used = PHOTO_SECONDS_FLOOR
                    floor_hit = True
                else:
                    used = max_seconds
    total_seconds = other_seconds + kept_count * used
    stills_share = (kept_count * used) / total_seconds if total_seconds > 0 else 0.0
    return {
        "requested_seconds": requested_seconds,
        "used_seconds": used,
        "stills_share": stills_share,
        "cap": STILLS_SHARE_CAP,
        "floor_hit": floor_hit,
    }


def max_stills_for_cap(*, other_seconds: float, floor_seconds: float = PHOTO_SECONDS_FLOOR) -> int:
    """The largest kept-still count whose share stays at or under
    ``STILLS_SHARE_CAP`` even at ``floor_seconds`` (the floor
    ``resolve_photo_seconds`` never shrinks below).

    Pure: given the seconds every other scene already takes, this is the
    count film.py trims kept stills DOWN TO before pacing, so
    ``resolve_photo_seconds``'s ``floor_hit`` becomes unreachable in a normal
    run instead of a warning-and-ship-anyway.
    """
    if other_seconds <= 0 or floor_seconds <= 0:
        return 0
    denominator = floor_seconds * (1 - STILLS_SHARE_CAP)
    if denominator <= 0:
        return 0
    count = max(0, int(math.floor((STILLS_SHARE_CAP * other_seconds) / denominator)) + 1)
    while count > 0:
        total_seconds = other_seconds + count * floor_seconds
        share = (count * floor_seconds) / total_seconds
        if share <= STILLS_SHARE_CAP:
            break
        count -= 1
    return count


def solve_map_seconds(
    *,
    target_seconds: float,
    kept_count: int,
    mapped_count: int,
    requested_photo_seconds: float,
    intro_seconds: float,
    recap_seconds: float,
    extra_seconds: float = 0.0,
) -> tuple[float, float]:
    """Map seconds (and the per-still seconds they imply) for a film of ``target_seconds``.

    The map is the elastic scene: intro, recap, every kept still and each
    mapped beat's settle hold keep their own seconds, and the map legs share
    what is left. Per-still seconds are the ones ``resolve_photo_seconds``
    picks for that map, so the pacing cap still holds -- solved in closed
    form, because a longer map lets stills run longer and vice versa.
    ``extra_seconds`` is any other fixed runtime (a placed source video's
    clip plus its arrival hold). Raises when the fixed scenes alone already
    fill the target.
    """
    held = mapped_count * MAP_SETTLE_SECONDS + extra_seconds
    fixed = intro_seconds + recap_seconds + held

    def paced(map_seconds: float) -> float:
        return float(
            resolve_photo_seconds(
                requested_seconds=requested_photo_seconds,
                kept_count=kept_count,
                intro_seconds=intro_seconds,
                map_seconds=map_seconds,
                recap_seconds=recap_seconds,
            )["used_seconds"]
        )

    map_seconds = target_seconds - fixed - kept_count * requested_photo_seconds
    if map_seconds <= 0 or paced(map_seconds) < requested_photo_seconds:
        # The cap binds: stills get exactly STILLS_SHARE_CAP of the runtime
        # that is not settle holds, never below the floor.
        capped = (
            STILLS_SHARE_CAP * (target_seconds - held) / kept_count
            if kept_count
            else requested_photo_seconds
        )
        photo_seconds = max(PHOTO_SECONDS_FLOOR, min(requested_photo_seconds, capped))
        map_seconds = target_seconds - fixed - kept_count * photo_seconds
    if map_seconds <= 0:
        raise ValueError(
            f"target {target_seconds:.1f}s is filled before any map plays: intro, recap, "
            f"{kept_count} stills, {mapped_count} settle holds and {extra_seconds:.1f}s "
            "of video need more; "
            "raise the target or keep fewer photos"
        )
    return map_seconds, paced(map_seconds)
