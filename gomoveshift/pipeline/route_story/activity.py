"""Route Story explicit activity contract.

Resolves what kind of trip a route story is -- hiking, cycling, or driving
-- into one shared :class:`ActivityProfile` that downstream surfaces read
instead of each independently re-guessing from speed. It is wired into ``route_story/film.py``'s display speed filter and
``route_story/metadata.py``'s derived title/tags. ``hud_speed_cap_mps`` and
``grade_cap_percent`` set the HUD's speed/grade display caps for the whole
render (``route_story/hud.py``'s ``set_active_display_caps``);
``default_units`` is carried, not yet consumed.

An explicit ``--activity``/profile-yaml value always wins. Absent one
(``None`` or ``"auto"``), the profile falls back to the same speed-band
guess ``route_story/metadata.py`` has always used for its own
``_activity_for_speed`` (kept in sync here, not re-derived, so "auto"
resolves identically wherever it's asked).

This module also resolves **display units**: an explicit
``--units``/profile-yaml value wins outright; absent one, the units
default from the route's own region (``resolve_units``). Bump
:data:`MAPPING_VERSION` whenever ``ACTIVITY_PROFILES``, the region
table, or the units rule changes, so a receipt can tell which table
produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import RoutePoint

HIKING_MAX_SPEED_MPS = 3.5
RIDING_MAX_SPEED_MPS = 9.0

#: Bump whenever ACTIVITY_PROFILES, IMPERIAL_REGIONS, or the units rule
#: changes, so a receipt can distinguish a re-render under a different table.
MAPPING_VERSION = "2026-09-25.1"

#: (name, lat_min, lat_max, lon_min, lon_max) bounding boxes for regions
#: whose everyday road/trail units are imperial. Everything else (Ireland
#: included) defaults to metric.
IMPERIAL_REGIONS: tuple[tuple[str, float, float, float, float], ...] = (
    ("continental_us", 24.5, 49.5, -125.0, -66.5),
    ("alaska", 51.0, 71.5, -180.0, -129.0),
    ("hawaii", 18.5, 22.5, -160.5, -154.5),
    ("puerto_rico", 17.8, 18.6, -67.4, -65.2),
    # Western edge trimmed to -5.5 (not the full -8.7 of Northern Ireland) so
    # this box does not also cover the Republic of Ireland, which shares its
    # longitude range but uses metric.
    ("uk", 49.8, 61.0, -5.5, 1.9),
)


@dataclass(frozen=True)
class ActivityProfile:
    """A route story's activity contract -- everything downstream keys to it."""

    name: str
    max_speed_mps: float
    hud_speed_cap_mps: float
    grade_cap_percent: float
    default_units: str


ACTIVITY_PROFILES: dict[str, ActivityProfile] = {
    "hiking": ActivityProfile(
        name="hiking",
        max_speed_mps=HIKING_MAX_SPEED_MPS,
        hud_speed_cap_mps=4.0,
        grade_cap_percent=35.0,
        default_units="imperial",
    ),
    "cycling": ActivityProfile(
        name="cycling",
        max_speed_mps=20.0,
        hud_speed_cap_mps=25.0,
        grade_cap_percent=20.0,
        default_units="imperial",
    ),
    "driving": ActivityProfile(
        name="driving",
        max_speed_mps=50.0,
        hud_speed_cap_mps=60.0,
        grade_cap_percent=12.0,
        default_units="imperial",
    ),
}


def _activity_name_for_speed(observed_max_speed_mps: float | None) -> str:
    if observed_max_speed_mps is None:
        return "driving"
    if observed_max_speed_mps <= HIKING_MAX_SPEED_MPS:
        return "hiking"
    if observed_max_speed_mps <= RIDING_MAX_SPEED_MPS:
        return "cycling"
    return "driving"


def resolve_activity(explicit: str | None, observed_max_speed_mps: float | None) -> ActivityProfile:
    """Resolve the activity contract for a route story.

    ``explicit`` (a ``--activity``/profile-yaml value) wins outright when it
    names a known activity; ``None`` or ``"auto"`` falls back to the
    speed-band guess above. Raises :class:`ValueError` for an unknown
    explicit name.
    """
    if explicit and explicit != "auto":
        try:
            return ACTIVITY_PROFILES[explicit]
        except KeyError:
            raise ValueError(f"unknown activity: {explicit!r}") from None
    return ACTIVITY_PROFILES[_activity_name_for_speed(observed_max_speed_mps)]


def region_units(latitude: float, longitude: float) -> str:
    """ "imperial" inside a known imperial-units region, else "metric"."""
    for _name, lat_min, lat_max, lon_min, lon_max in IMPERIAL_REGIONS:
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return "imperial"
    return "metric"


def resolve_units(explicit: str | None, centroid: tuple[float, float] | None) -> tuple[str, str]:
    """Resolve display units for a route story.

    ``explicit`` (a ``--units``/profile-yaml value) wins outright when it is
    ``"metric"`` or ``"imperial"``. ``None`` or ``"auto"`` falls back to the
    track's region (``region_units`` on ``centroid``), or metric when no
    centroid is available. Raises :class:`ValueError` for an unknown
    explicit value.
    """
    if explicit in ("metric", "imperial"):
        return explicit, "explicit"
    if explicit not in (None, "auto"):
        raise ValueError(f"unknown units: {explicit!r}")
    if centroid is None:
        return "metric", "default"
    return region_units(*centroid), "region"


def track_centroid(
    segments: Sequence[Sequence["RoutePoint"]],
) -> tuple[float, float] | None:
    """Mean (latitude, longitude) over every point in every segment, or None."""
    latitudes = [point.latitude for segment in segments for point in segment]
    longitudes = [point.longitude for segment in segments for point in segment]
    if not latitudes:
        return None
    return sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)
