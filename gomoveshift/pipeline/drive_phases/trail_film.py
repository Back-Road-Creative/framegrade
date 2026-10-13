"""Route Story inside a drive: one trail film per LF day, a sibling deliverable.

``attach_trail_film`` runs after a day's LFs render. It resolves the day's GPX
and the prepared imagery bundle (the "prepare" step is a lookup: the network
fetch never runs inside a drive), renders a terrain proof clip, verifies its
receipt, then renders the film with the day's first LF as the placed source
clip. The result lands on the day's ``Deliverables`` (``trail_films``); a day
that cannot get a film records why in ``trail_skips`` and logs a warning. A
trail failure never fails the drive: the LFs are the primary product.

Before any proof or film encode, a measured preflight (``route_story.preflight.
probe_3d_renderer``) proves the 3D terrain renderer works: Playwright (the
``route-story-3d`` extra) importable, the pinned Chromium launching, and the day's
bundle's terrain scene drawing ready. A host that fails it records a skip naming
the cause and runs no encode; under an explicit ``--trail`` it is logged as an error.

``config.drive.trail`` is tri-state: ``None`` (auto, on when the day has a GPX
and a prepared bundle), ``True`` (``--trail``) or ``False`` (``--no-trail``, a
silent no-op). Imagery rasters and credits come from the Route Story launch
profile named by ``config.drive.trail_profile`` (``--trail-profile``); the day's
own GPX, date, source clip, output and photos override whatever the profile holds.

The film gets only the day's photos. ``film.main`` refuses a photos dir holding
media it cannot place on the route, and a drive's photo folder spans every day,
so the day's photos (capture clock inside its GPX span, widened by the film's own
bookend window) are linked into ``trail/photos_<date>`` and that is what
``--photos-dir`` points at. A day recorded with a trail skip is re-run on resume
by :func:`retry_trail_skip`, trail step only, once its cause is resolved.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from ..media_api import Deliverables, TrailFilm
from ..route_story.preflight import TerrainRendererUnavailableError

logger = logging.getLogger(__name__)

#: The proof clip is a cheap gate before the long film render, not a deliverable.
PROOF_WIDTH, PROOF_HEIGHT, PROOF_FPS = 960, 540, 24.0


#: The stills ``film.main`` reads from ``--photos-dir`` (its own inline set).
PHOTO_SUFFIXES = frozenset({".nef", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"})


def _gpx_span(gpx: Path, recorded_on: _date) -> tuple[datetime, datetime]:
    """UTC span of the day's GPX track, selected exactly as ``film.main`` selects it."""
    from ..route_story.gpx import select_gpx_day

    selection = select_gpx_day(gpx, recorded_on)
    offsets = [point.time_s for segment in selection.segments for point in segment]
    start = selection.source_started_at
    return start + timedelta(seconds=min(offsets)), start + timedelta(seconds=max(offsets))


def _capture_utc(photo: Path) -> Optional[datetime]:
    """The photo's zoned EXIF capture instant (the clock ``film.main`` places it by)."""
    from ..photos.exif_gps import extract_exif_capture_metadata

    return extract_exif_capture_metadata(photo).capture_utc


@dataclass(frozen=True)
class TrailSteps:
    """The Route Story steps, injectable so the wiring is testable.

    ``gpx_span`` and ``capture_utc`` place a photo on the day's clock.
    """

    find_bundle: Callable[..., Optional[Path]]
    render_proof: Callable[..., Path]
    verify: Callable[[Path], Any]
    film: Callable[[list], int]
    #: Raises ``TerrainRendererUnavailableError`` unless the 3D renderer draws the bundle.
    probe_3d: Callable[[Path], None]
    gpx_span: Callable[[Path, _date], tuple[datetime, datetime]] = _gpx_span
    capture_utc: Callable[[Path], Optional[datetime]] = _capture_utc


def default_steps() -> TrailSteps:
    from ..route_story import film, preflight, prepare, proof, verify

    return TrailSteps(
        find_bundle=prepare.find_prepared_bundle,
        render_proof=lambda gpx, bundle, output, *, recorded_on: proof.render_proof_with_receipt(
            gpx,
            bundle,
            output,
            recorded_on=recorded_on,
            fps=PROOF_FPS,
            width=PROOF_WIDTH,
            height=PROOF_HEIGHT,
        ),
        verify=verify.verify_receipt,
        film=film.main,
        probe_3d=preflight.probe_3d_renderer,
    )


class _SkipError(Exception):
    """A day that cannot get a trail film; the message is the recorded reason."""


class _RendererRefusedError(_SkipError):
    """The 3D renderer preflight failed; ``--trail`` makes this an error, auto a warning."""


def _day_gpx(state: Any, day: str) -> Path:
    dates = getattr(state, "episode_dates", None) or []
    gpx_files = getattr(state, "gpx_files", None) or []
    for i, episode_date in enumerate(dates):
        if str(episode_date) == day and i < len(gpx_files) and gpx_files[i] is not None:
            return Path(gpx_files[i])
    raise _SkipError(f"no GPX for {day}")


def _first_lf(d: Deliverables) -> tuple[int, Path]:
    pairs = sorted(zip(d.regular_lf_indices, d.regular_lf_paths), key=lambda p: p[0])
    if not pairs:
        raise _SkipError(f"no regular LF rendered for {d.date}")
    return int(pairs[0][0]), Path(pairs[0][1])


def _stage_day_photos(pool: Path, stage: Path, gpx: Path, day: str, steps: TrailSteps) -> Path:
    """Link the day's photos from ``pool`` into ``stage`` and return ``stage``.

    A photo is the day's when its capture instant falls inside the day's GPX span,
    widened by ``film.BOOKEND_MAX_OUTSIDE_S`` so the film's own same-day bookend
    photos still reach it. A photo with no zoned capture time cannot be placed on
    the day's clock (``film.main`` would abort on it), so it is left out, loudly.
    """
    from ..route_story.film import BOOKEND_MAX_OUTSIDE_S

    if not pool.is_dir():
        raise _SkipError(f"photos dir {pool} not found")
    recorded_on = _date.fromisoformat(day)
    try:
        start, end = steps.gpx_span(gpx, recorded_on)
    except ValueError as exc:
        raise _SkipError(f"cannot read the GPX span of {gpx.name} for {day}: {exc}") from exc
    margin = timedelta(seconds=BOOKEND_MAX_OUTSIDE_S)
    if stage.is_dir():  # a retry re-stages from scratch, never on top of stale links
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    undated: list[str] = []
    for photo in sorted(pool.iterdir()):
        if photo.suffix.lower() not in PHOTO_SUFFIXES or not photo.is_file():
            continue
        taken = steps.capture_utc(photo)
        if taken is None:
            undated.append(photo.name)
        elif start - margin <= taken <= end + margin:
            try:
                os.symlink(photo.resolve(), stage / photo.name)
            except OSError:
                shutil.copy2(photo, stage / photo.name)
    if undated:
        logger.warning(
            "  trail: %d photo(s) in %s have no zoned capture time and cannot be placed on %s: %s",
            len(undated),
            pool,
            day,
            ", ".join(undated),
        )
    return stage


def _film_argv(
    profile_path: Path,
    overrides: dict,
    source_dir: Any,
    stage_photos: Callable[[Path], Path],
) -> list[str]:
    from ..route_story.profile import (
        ProfileError,
        build_film_argv,
        film_parser,
        load_profile,
        unverified_values,
    )

    try:
        profile = load_profile(profile_path)
        # The pool is the profile's photos-dir, else the drive's source dir; the
        # film is handed only the slice of it that belongs to this day.
        profile["photos-dir"] = str(stage_photos(Path(profile.get("photos-dir") or source_dir)))
        profile.update(overrides)
        argv = build_film_argv(film_parser(), profile)
    except ProfileError as exc:
        raise _SkipError(f"trail profile {profile_path}: {exc}") from exc
    unverified = unverified_values(argv)
    if unverified:
        raise _SkipError(f"trail profile {profile_path} has unverified value(s): {unverified}")
    return argv


@dataclass(frozen=True)
class _DayInputs:
    gpx: Path
    lf_index: int
    lf_path: Path
    bundle: Path
    profile: Path


def _resolve_inputs(d: Deliverables, state: Any, config: Any, steps: TrailSteps) -> _DayInputs:
    """Everything a day's film needs that already exists; a missing one is a skip."""
    day = str(d.date)
    gpx = _day_gpx(state, day)
    lf_index, lf_path = _first_lf(d)
    try:
        bundle = steps.find_bundle(gpx, recorded_on=_date.fromisoformat(day))
    except (ValueError, OSError) as exc:
        raise _SkipError(f"cannot resolve a bundle for {gpx.name} on {day}: {exc}") from exc
    if bundle is None:
        raise _SkipError(
            f"no prepared imagery bundle for {gpx.name} on {day}; "
            f"run `gomoveshift route-story-prepare {gpx} --recorded-on {day}` first"
        )
    profile = str(getattr(config.drive, "trail_profile", "") or "")
    if not profile:
        raise _SkipError("no --trail-profile (the film needs imagery rasters and credits)")
    return _DayInputs(gpx, lf_index, lf_path, bundle, Path(profile))


def _render(d: Deliverables, state: Any, config: Any, steps: TrailSteps) -> TrailFilm:
    day = str(d.date)
    inputs = _resolve_inputs(d, state, config, steps)
    gpx, lf_index, lf_path, bundle = inputs.gpx, inputs.lf_index, inputs.lf_path, inputs.bundle
    recorded_on = _date.fromisoformat(day)

    out_dir = lf_path.parent.parent / "trail"
    proof_path = out_dir / f"trail_proof_{day}.mp4"
    film_path = out_dir / f"trail_film_{day}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Measured before any encode: the film is 3D terrain, so a host that cannot
    # launch the browser and draw a terrain tile refuses here, not hours in.
    try:
        steps.probe_3d(bundle)
    except TerrainRendererUnavailableError as exc:
        raise _RendererRefusedError(
            f"no working 3D terrain renderer ({exc}); install the 'route-story-3d' extra "
            f"and set GMS_ROUTE_STORY_CHROMIUM"
        ) from exc

    receipt = steps.render_proof(gpx, bundle, proof_path, recorded_on=recorded_on)
    report = steps.verify(receipt)
    if not report.ok:
        failed = "; ".join(f"{c.name}: {c.detail}" for c in report.checks if not c.ok)
        raise _SkipError(f"proof receipt failed verification ({failed or 'no checks ran'})")

    argv = _film_argv(
        inputs.profile,
        {
            "gpx": str(gpx),
            "recorded-on": day,
            "source-video": str(lf_path),
            "output": str(film_path),
            "receipt": str(out_dir / f"trail_film_{day}.receipt.json"),
            "renderer": "scene",
            # The bundle the proof just rendered and verified, not a second lookup.
            "bundle": str(bundle),
        },
        getattr(state, "source_dir", "."),
        lambda pool: _stage_day_photos(pool, out_dir / f"photos_{day}", gpx, day, steps),
    )
    try:
        rc = steps.film(argv)
    except SystemExit as exc:  # film.main reports bad inputs through parser.error
        rc = exc.code if isinstance(exc.code, int) else 2
    if rc != 0:
        raise _SkipError(f"film render failed (exit {rc})")
    if not film_path.is_file():
        raise _SkipError(f"film render reported success but wrote no {film_path.name}")
    return TrailFilm(
        lf_index=lf_index,
        path=film_path,
        receipt=Path(argv[argv.index("--receipt") + 1]),
    )


def attach_trail_film(
    d: Deliverables, state: Any, config: Any, *, steps: Optional[TrailSteps] = None
) -> Deliverables:
    """Render ``d``'s trail film in place; a day that cannot get one is recorded."""
    if getattr(getattr(config, "drive", None), "trail", None) is False:
        return d
    try:
        d.trail_films.append(_render(d, state, config, steps or default_steps()))
    except _SkipError as skip:
        reason = f"trail film skipped for {d.date}: {skip}"
        explicit = getattr(getattr(config, "drive", None), "trail", None) is True
        if isinstance(skip, _RendererRefusedError) and explicit:
            logger.error("  trail: %s", reason)  # --trail asked for the film
        else:
            logger.warning("  trail: %s", reason)
        d.trail_skips.append(reason)
    except Exception as exc:  # noqa: BLE001 -- a sibling deliverable never fails the drive
        reason = f"trail film failed for {d.date}: {type(exc).__name__}: {exc}"
        logger.error("  trail: %s", reason, exc_info=True)
        d.trail_skips.append(reason)
    return d


def retry_trail_skip(
    d: Deliverables, state: Any, config: Any, *, steps: Optional[TrailSteps] = None
) -> bool:
    """Re-run only the trail step for a resumed day recorded with a skip.

    The skip is persisted with the day, so without this a resume would carry it
    forever. A day is retried when it has recorded skips, no film, and every input
    a film needs now exists (GPX, rendered LF, prepared bundle, ``--trail-profile``);
    until then the recorded skip stands and nothing runs. A retry replaces the old
    skips with its own outcome and never re-runs the day. Returns whether it ran.
    """
    if getattr(getattr(config, "drive", None), "trail", None) is False:
        return False
    if d.trail_films or not d.trail_skips:
        return False
    steps = steps or default_steps()
    try:
        _resolve_inputs(d, state, config, steps)
    except _SkipError:
        return False
    d.trail_skips = []
    attach_trail_film(d, state, config, steps=steps)
    return True
