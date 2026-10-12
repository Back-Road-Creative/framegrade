"""Editorial presets for the Route Story film: ``full`` (16:9) vs ``social`` (9:16).

A preset supplies DEFAULTS only -- ``film.main`` fills an unset CLI flag from
the chosen preset, and any explicit ``--width``/``--height``/``--photo-seconds``/
``--map-seconds`` the operator passes always wins. ``full`` reproduces the
service's pre-preset defaults exactly (see ``test_route_story_presets.py``),
so an operator who never passes ``--preset`` sees byte-identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilmPreset:
    """One editorial shape: delivery size, pacing, an optional length ceiling.

    ``max_photos`` caps the kept-still count on top of the existing
    ``still_selection.max_stills_for_cap`` share cap -- whichever is smaller
    wins. ``max_seconds``, when set, is enforced BEFORE any render starts
    (``film.main`` refuses a plan that would exceed it), never after an
    encode. ``rendition`` names the delivery rendition this preset targets;
    ``film.main`` never re-encodes for it -- ``encode_final`` is the only
    delivery encode -- but does verify, after the fact, that the delivered
    file actually matches that rendition (``receipt["preset"]["rendition_check"]``).
    """

    name: str
    width: int
    height: int
    intro_seconds: float
    recap_seconds: float
    photo_seconds: float
    map_seconds: float | None
    max_photos: int | None
    max_seconds: float | None
    rendition: str | None


# These mirror film.py's pre-preset literals (DELIVERY_WIDTH/HEIGHT,
# DEFAULT_INTRO_SECONDS/RECAP_SECONDS, and the historical --photo-seconds/
# --map-seconds CLI defaults of 2.5/None) exactly -- verified equal in
# test_route_story_presets.py rather than imported, to avoid a film.py <->
# presets.py import cycle (film.py imports PRESETS).
FULL = FilmPreset(
    name="full",
    width=1920,
    height=1080,
    intro_seconds=12.0,
    recap_seconds=6.0,
    photo_seconds=2.5,
    map_seconds=None,
    max_photos=None,
    max_seconds=None,
    rendition=None,
)

# Editorial numbers for a short-form 9:16 cut with a tight opening, fewer stills, a compressed map
# leg, and a hard 90s ceiling enforced upstream of any render.
SOCIAL = FilmPreset(
    name="social",
    width=1080,
    height=1920,
    intro_seconds=6.0,
    recap_seconds=4.0,
    photo_seconds=2.0,
    map_seconds=27.0,
    max_photos=8,
    max_seconds=90.0,
    # Audio: this cut keeps its attenuate-only audio --
    # it is not cabin audio, so it does NOT take "youtube_short", the
    # drive-shorts rendition that ships -an.
    rendition="route_story_short",
)

PRESETS: dict[str, FilmPreset] = {"full": FULL, "social": SOCIAL}


class FilmTooLongError(ValueError):
    """A preset's plan exceeds its own length ceiling -- refused before any render."""


def apply_preset(
    preset: FilmPreset,
    *,
    width: int | None,
    height: int | None,
    photo_seconds: float | None,
    map_seconds: float | None,
) -> tuple[int, int, float, float | None]:
    """Fill an unset (``None``) CLI value from ``preset``; an explicit value always wins.

    ``None`` is unambiguous here: none of these flags accepts a way to
    request ``None`` from the command line, so post-parse ``None`` can only
    mean "the operator didn't pass this flag".
    """
    return (
        width if width is not None else preset.width,
        height if height is not None else preset.height,
        photo_seconds if photo_seconds is not None else preset.photo_seconds,
        map_seconds if map_seconds is not None else preset.map_seconds,
    )


def capped_still_count(preset: FilmPreset, *, cap_limit: int) -> int:
    """The lower of the share-based ``cap_limit`` and the preset's own ``max_photos``."""
    if preset.max_photos is None:
        return cap_limit
    return min(cap_limit, preset.max_photos)


def check_ceiling(preset: FilmPreset, *, planned_seconds: float) -> None:
    """Refuse a plan over ``preset.max_seconds`` -- a no-op when the preset sets none."""
    if preset.max_seconds is not None and planned_seconds > preset.max_seconds:
        raise FilmTooLongError(
            f"--preset {preset.name}: planned film is {planned_seconds:.1f}s, over "
            f"the {preset.max_seconds:.0f}s ceiling"
        )
