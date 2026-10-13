"""Route Story inside a drive: one trail film per LF day, a sibling deliverable.

``attach_trail_film`` runs after a day's LFs render. It resolves the day's GPX
and the prepared imagery bundle (the "prepare" step is a lookup: the network
fetch never runs inside a drive), renders a terrain proof clip, verifies its
receipt, then renders the film with the day's first LF as the placed source
clip. The result lands on the day's ``Deliverables`` (``trail_films``); a day
that cannot get a film records why in ``trail_skips`` and logs a warning. A
trail failure never fails the drive: the LFs are the primary product.

``config.drive.trail`` is tri-state: ``None`` (auto, on when the day has a GPX
and a prepared bundle), ``True`` (``--trail``) or ``False`` (``--no-trail``, a
silent no-op). Imagery rasters and credits come from the Route Story launch
profile named by ``config.drive.trail_profile`` (``--trail-profile``); the day's
own GPX, date, source clip and output override whatever the profile holds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable, Optional

from ..media_api import Deliverables, TrailFilm

logger = logging.getLogger(__name__)

#: The proof clip is a cheap gate before the long film render, not a deliverable.
PROOF_WIDTH, PROOF_HEIGHT, PROOF_FPS = 960, 540, 24.0


@dataclass(frozen=True)
class TrailSteps:
    """The four Route Story steps, injectable so the wiring is testable."""

    find_bundle: Callable[..., Optional[Path]]
    render_proof: Callable[..., Path]
    verify: Callable[[Path], Any]
    film: Callable[[list], int]


def default_steps() -> TrailSteps:
    from ..route_story import film, prepare, proof, verify

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
    )


class _SkipError(Exception):
    """A day that cannot get a trail film; the message is the recorded reason."""


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


def _film_argv(profile_path: Path, overrides: dict, source_dir: Any) -> list[str]:
    from ..route_story.profile import (
        ProfileError,
        build_film_argv,
        film_parser,
        load_profile,
        unverified_values,
    )

    try:
        profile = load_profile(profile_path)
        profile.setdefault("photos-dir", str(source_dir))
        profile.update(overrides)
        argv = build_film_argv(film_parser(), profile)
    except ProfileError as exc:
        raise _SkipError(f"trail profile {profile_path}: {exc}") from exc
    unverified = unverified_values(argv)
    if unverified:
        raise _SkipError(f"trail profile {profile_path} has unverified value(s): {unverified}")
    return argv


def _render(d: Deliverables, state: Any, config: Any, steps: TrailSteps) -> TrailFilm:
    day = str(d.date)
    gpx = _day_gpx(state, day)
    lf_index, lf_path = _first_lf(d)
    recorded_on = _date.fromisoformat(day)
    try:
        bundle = steps.find_bundle(gpx, recorded_on=recorded_on)
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

    out_dir = lf_path.parent.parent / "trail"
    proof_path = out_dir / f"trail_proof_{day}.mp4"
    film_path = out_dir / f"trail_film_{day}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    receipt = steps.render_proof(gpx, bundle, proof_path, recorded_on=recorded_on)
    report = steps.verify(receipt)
    if not report.ok:
        failed = "; ".join(f"{c.name}: {c.detail}" for c in report.checks if not c.ok)
        raise _SkipError(f"proof receipt failed verification ({failed or 'no checks ran'})")

    argv = _film_argv(
        Path(profile),
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
        logger.warning("  trail: %s", reason)
        d.trail_skips.append(reason)
    except Exception as exc:  # noqa: BLE001 -- a sibling deliverable never fails the drive
        reason = f"trail film failed for {d.date}: {type(exc).__name__}: {exc}"
        logger.error("  trail: %s", reason, exc_info=True)
        d.trail_skips.append(reason)
    return d
