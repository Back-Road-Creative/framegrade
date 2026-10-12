"""A real sun for the Route Story terrain scene, replacing a constant light.

A fixed light azimuth with a constant hillshade exaggeration would double-light
NAIP imagery's own midday shadows and puts the highlight on the wrong side of
every ridge whenever the route wasn't actually shot facing northwest at dusk.
This module is the light MODEL only — a pure-Python, dependency-light
computation of where the sun actually was at a given place and instant, and
how strongly the terrain mesh should be relief-shaded at a given zoom. ``proof.py``
and ``scene_renderer.py`` pass a per-frame :class:`SceneLight` built from
these results to ``scene.js`` (``setLight``); this module is the "real sun"
model and nothing downstream of it.

Elevation comes from :func:`sun_track.solar_elevation_deg` (apparent, with
atmospheric refraction). ``sun_track`` exposes no azimuth, so
:func:`_solar_azimuth_deg` below implements the standard NOAA/Duffie-Beckman
solar-position formula: the same fractional-year Fourier fit for equation of
time and declination that ``sun_track`` uses internally to get an hour angle,
then the horizontal-coordinate conversion

    sin(alt)      = sin(lat) sin(decl) + cos(lat) cos(decl) cos(H)
    sin(gamma_s)  = cos(decl) sin(H) / cos(alt)
    cos(gamma_s)  = (sin(alt) sin(lat) - sin(decl)) / (cos(alt) cos(lat))

(Duffie & Beckman, *Solar Engineering of Thermal Processes*), where H is the
hour angle (negative before solar noon, positive after) and gamma_s is the
solar azimuth measured from due south, positive toward the west. Converting
to a compass bearing (clockwise from north, what MapLibre's
``hillshade-illumination-direction`` wants) is ``(180 + gamma_s) mod 360``.

Deterministic, standard-library ``math`` only. No network, no file I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sun_track import solar_elevation_deg

__all__ = [
    "SceneLight",
    "hillshade_exaggeration",
    "light_at",
    "light_for_pose",
    "sun_sweep",
]

# Civil twilight: below this apparent solar elevation, the scene reads as
# "night" and gets a low, fixed light rather than an ever-dimmer real sun —
# see SceneLight's docstring for why that beats inventing moonlight.
_NIGHT_ELEVATION_DEG = -6.0

# Intensity floor (night, and the dusk/dawn band the sun clamp shares with
# it) and the elevation at which the day-side ramp reaches full brightness.
_INTENSITY_FLOOR = 0.15
_INTENSITY_FULL_ELEVATION_DEG = 45.0


def _require_aware(when: datetime, *, arg_name: str) -> datetime:
    """``when`` normalized to UTC, or ``ValueError`` if it is naive.

    Mirrors ``sun_track.solar_elevation_deg``'s own contract: a wall-clock
    time silently read as UTC can be hours off, and an hour of hour-angle is
    10-18 degrees of elevation at typical latitudes. Resolving the offset is
    the caller's job, not a guess this module makes for them.
    """
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise ValueError(f"{arg_name} requires a tz-aware datetime; resolve it to UTC first")
    return when.astimezone(timezone.utc)


def _fractional_year_terms(utc: datetime) -> tuple[float, float]:
    """(equation_of_time_minutes, declination_rad) — the NOAA Fourier fit.

    Identical to the private computation inside ``sun_track.solar``, which
    exports elevation but not these intermediate terms nor an azimuth. Kept
    here rather than imported because nothing in ``sun_track``'s public API
    exposes it.
    """
    year = utc.year
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    frac_hour = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = (
        2 * math.pi / (366 if leap else 365) * (utc.timetuple().tm_yday - 1 + (frac_hour - 12) / 24)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    return eqtime, decl


def _solar_azimuth_deg(lat: float, lon: float, utc: datetime) -> float:
    """Compass bearing (degrees clockwise from north) of the sun at ``utc``.

    Geometric (no refraction correction — refraction shifts apparent
    elevation by at most a fraction of a degree near the horizon and has no
    defined effect on azimuth). See the module docstring for the formula and
    its citation.
    """
    eqtime, decl = _fractional_year_terms(utc)
    frac_hour = utc.hour + utc.minute / 60 + utc.second / 3600
    true_solar_min = frac_hour * 60 + eqtime + 4 * lon
    hour_angle = math.radians(true_solar_min / 4 - 180)

    lat_rad = math.radians(lat)
    sin_alt = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(
        hour_angle
    )
    sin_alt = max(-1.0, min(1.0, sin_alt))
    cos_alt = math.sqrt(1.0 - sin_alt * sin_alt)

    cos_lat = math.cos(lat_rad)
    if cos_alt < 1e-9 or abs(cos_lat) < 1e-9:
        # Sun directly overhead, or an observer at a pole: azimuth is
        # undefined. South (180 deg) is as good a fallback as any, and this
        # only happens within a hair of the subsolar point or the poles.
        return 180.0

    sin_az = math.cos(decl) * math.sin(hour_angle) / cos_alt
    cos_az = (sin_alt * math.sin(lat_rad) - math.sin(decl)) / (cos_alt * cos_lat)
    gamma_s = math.atan2(sin_az, cos_az)  # from due south, positive toward west
    return (180.0 + math.degrees(gamma_s)) % 360.0


@dataclass(frozen=True)
class SceneLight:
    """One MapLibre scene light: direction, strength, and day/night state.

    ``azimuth_deg`` is clockwise from north (MapLibre's
    ``hillshade-illumination-direction`` convention). ``elevation_deg`` is
    never negative — MapLibre has no use for a light below the horizon, and a
    negative value there is a bug waiting to render a lit night side. Nights
    (apparent solar elevation below civil twilight, -6 degrees) get a low,
    fixed ``intensity`` instead of inventing moonlight: nothing here claims to
    model the moon.
    """

    azimuth_deg: float
    elevation_deg: float
    intensity: float
    night: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.azimuth_deg < 360.0:
            raise ValueError("azimuth_deg must be in [0, 360)")
        if self.elevation_deg < 0.0:
            raise ValueError("elevation_deg must be non-negative")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be in [0, 1]")

    def to_scene(self, zoom: float) -> dict[str, float | bool]:
        """The dict ``scene.js``'s ``setLight`` reads for this light at ``zoom``.

        Night keeps whatever (already low) ``intensity`` was computed here —
        nothing on the scene side is allowed to invent moonlight.
        """
        return {
            "azimuth": self.azimuth_deg,
            "elevation": self.elevation_deg,
            "intensity": self.intensity,
            "night": self.night,
            "hillshadeExaggeration": hillshade_exaggeration(zoom),
        }


def light_at(lat: float, lon: float, when: datetime) -> SceneLight:
    """The scene light for the sun's real position at (``lat``, ``lon``, ``when``).

    ``when`` must be tz-aware; a naive value raises ``ValueError`` rather than
    guessing a zone.
    """
    utc = _require_aware(when, arg_name="when")
    raw_elevation = solar_elevation_deg(lat, lon, utc)
    azimuth = _solar_azimuth_deg(lat, lon, utc)
    night = raw_elevation < _NIGHT_ELEVATION_DEG

    elevation = max(0.0, raw_elevation)
    ramp = max(0.0, min(1.0, elevation / _INTENSITY_FULL_ELEVATION_DEG))
    intensity = _INTENSITY_FLOOR + (1.0 - _INTENSITY_FLOOR) * ramp
    if night:
        intensity = _INTENSITY_FLOOR

    return SceneLight(
        azimuth_deg=azimuth, elevation_deg=elevation, intensity=intensity, night=night
    )


def hillshade_exaggeration(zoom: float) -> float:
    """Relief exaggeration for the hillshade layer at ``zoom``.

    Linear from 0.4 at zoom <= 12 down to 0.0 at zoom >= 14. At low zoom the
    DEM mesh is too coarse for relief to read without help; by zoom 14 the
    mesh plus imagery carry it on their own, and any exaggeration on top just
    re-adds the midday shadows NAIP's own imagery already bakes in.
    """
    if zoom <= 12.0:
        return 0.4
    if zoom >= 14.0:
        return 0.0
    return 0.4 * (14.0 - zoom) / 2.0


def sun_sweep(
    lat: float, lon: float, start: datetime, end: datetime, frames: int
) -> list[SceneLight]:
    """``frames`` scene lights evenly spaced in time from ``start`` to ``end``.

    For the time-lapse closing pullback: deterministic, ``frames`` clock
    instants linearly spaced (inclusive of both ends), so the first and last
    entries are exactly ``light_at(lat, lon, start)`` / ``light_at(lat, lon,
    end)``. Needs at least two frames — a "sweep" of one instant isn't one.
    """
    if frames < 2:
        raise ValueError("sun_sweep needs at least two frames")
    start_utc = _require_aware(start, arg_name="start")
    end_utc = _require_aware(end, arg_name="end")
    step = (end_utc - start_utc) / (frames - 1)
    return [light_at(lat, lon, start_utc + step * i) for i in range(frames)]


def light_for_pose(
    pose_time: datetime | None, lat: float, lon: float, *, fallback: datetime
) -> SceneLight:
    """The light for one camera pose: its own route time, or ``fallback``.

    Chase/follow frames carry the route's own clock time (``pose_time``).
    Descent/pullback frames have no route time of their own and pass
    ``None``; the caller decides whether ``fallback`` is the trip's start or
    end instant.
    """
    return light_at(lat, lon, pose_time if pose_time is not None else fallback)
