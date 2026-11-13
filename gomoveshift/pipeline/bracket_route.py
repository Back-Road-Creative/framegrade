"""Bracket route: the on-screen label for a GPS-less drive, COMPUTED from the GPS
fixes recorded immediately before and after the footage (``--bracket-gpx BEFORE AFTER``).

The specificity ladder (``title_builder.award_tier``) names a bracket-route tier, but
until this module nothing fed it: the only evidence source was the LF's own GPS model,
so a drive whose camera recorded no fix could carry only a claim-free placeless label.

The label is never typed. It is the narrowest region that BOTH bracketing fixes
reverse-geocode to -- same county → ``"County, State"`` (or ``"County, Country"`` where
there is no state), else same state → the state, else two states of one country →
``"State A to State B"`` in travel order (the leg between two measured fixes, level
``states``), else same country → the country, else refused. The two fixes must be in time order and every footage recording date must
fall between them. How far apart they may be depends on the level claimed
(:data:`BRACKET_MAX_GAP_H_BY_LEVEL`): a county or state needs a tight bracket, a country
survives a longer one, and a gap past a level's cap drops the claim to the next broader
level the fixes agree on. Any failure is a
:class:`BracketRouteRefusedError`: a bracket that cannot prove a region names none.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.pipeline.gopro_gps import extract_gopro_gpx
from gomoveshift.pipeline.gps_enrichment import region_label

#: Longest gap (hours) between the last fix before the footage and the first fix after it
#: that still proves each level. A day track that stops the evening before and resumes the
#: morning after proves a county; a day off the track between two tracks (≈ 44 h) can no
#: longer prove a county, but a road trip does not leave the country in two days. A gap past
#: a level's cap names only a broader level the two fixes agree on.
BRACKET_MAX_GAP_H_BY_LEVEL = {"county": 36.0, "state": 36.0, "states": 36.0, "country": 48.0}
#: Level of a two-state leg label; it is claimed under the ``state`` cap.
STATES_LEVEL = "states"
#: Longest gap that brackets anything at all; past it the bracket is refused.
BRACKET_MAX_GAP_H = max(BRACKET_MAX_GAP_H_BY_LEVEL.values())
#: ``tier_reason`` recorded on an LF whose label is the bracket label.
BRACKET_TIER_REASON = "bracket_route"
#: Ladder tier awarded per common-region level: a county agreed by both fixes is the one
#: corroboration tier 3 asks for; a bare state (or a country, which is broader) is the
#: ladder's "state only -- bracket route alone" tier 4.
BRACKET_LEVEL_TIER = {"county": 3, "state": 4, "states": 4, "country": 4}
_PLACE_KEYS = ("county", "state", "country")
_UNRESOLVED = {"", "none", "null", "unknown"}
_GPX_SUFFIXES = {".gpx"}
_VIDEO_SUFFIXES = {".mp4", ".mov"}


def bracket_record_valid(record) -> bool:
    """``record`` (``state.bracket_route``) names a label at a known level and its fix
    gap is within that level's cap (:data:`BRACKET_MAX_GAP_H_BY_LEVEL`)."""
    if not isinstance(record, dict) or not str(record.get("label") or "").strip():
        return False
    cap = BRACKET_MAX_GAP_H_BY_LEVEL.get(str(record.get("level") or ""))
    try:
        gap = float(record["gap_h"])
    except (KeyError, TypeError, ValueError):
        return False
    return cap is not None and 0 <= gap <= cap


class BracketRouteRefusedError(ValueError):
    """The two files do not bracket this footage with one common region."""


@dataclass(frozen=True)
class BracketFix:
    lat: float
    lon: float
    ts: float  # Unix seconds
    source: str  # the file the fix came from, as given on the CLI

    @property
    def time_iso(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class BracketRoute:
    before: BracketFix
    after: BracketFix
    gap_h: float
    before_place: dict
    after_place: dict
    level: str
    label: str

    def to_record(self) -> dict:
        """JSON-scalar provenance record (``state.bracket_route``, the sidecar)."""
        return {
            "label": self.label,
            "level": self.level,
            "gap_h": round(float(self.gap_h), 3),
            "before": {**asdict(self.before), "time": self.before.time_iso},
            "after": {**asdict(self.after), "time": self.after.time_iso},
            "before_place": dict(self.before_place),
            "after_place": dict(self.after_place),
        }

    @classmethod
    def from_record(cls, rec: dict) -> "BracketRoute":
        def fix(d: dict) -> BracketFix:
            return BracketFix(float(d["lat"]), float(d["lon"]), float(d["ts"]), str(d["source"]))

        return cls(
            before=fix(rec["before"]),
            after=fix(rec["after"]),
            gap_h=float(rec["gap_h"]),
            before_place=dict(rec["before_place"]),
            after_place=dict(rec["after_place"]),
            level=str(rec["level"]),
            label=str(rec["label"]),
        )


def _slot(place: dict, key: str) -> str:
    value = str((place or {}).get(key) or "").strip()
    return "" if value.lower() in _UNRESOLVED else value


def common_region(
    before_place: dict, after_place: dict, levels: Iterable[str] = _PLACE_KEYS
) -> tuple[str, str]:
    """``(level, label)``: the narrowest region in ``levels`` both geocodes agree on.

    county → :func:`region_label` (``"County, State"``; a stateless county carries its
    country, the same tail the per-LF model uses); state → the state; two different
    states of one country → ``"<before state> to <after state>"`` (level ``states``:
    the leg between the two fixes, 2026-11-13 rec0063 "United States"); country → the
    country. Unresolved slots never match each other. A level left out of ``levels``
    (its gap cap is exceeded) is never claimed.
    """
    allowed = set(levels)
    b = {k: _slot(before_place, k) for k in _PLACE_KEYS}
    a = {k: _slot(after_place, k) for k in _PLACE_KEYS}
    same = {k: bool(b[k]) and b[k].casefold() == a[k].casefold() for k in _PLACE_KEYS}
    if (
        "county" in allowed
        and same["county"]
        and same["country"]
        and (same["state"] or not (b["state"] or a["state"]))
    ):
        label = (
            region_label(b["county"], b["state"])
            if b["state"]
            else f"{b['county']}, {b['country']}"
        )
        return "county", str(label)
    if "state" in allowed and same["state"] and same["country"]:
        return "state", b["state"]
    if "state" in allowed and same["country"] and b["state"] and a["state"]:
        return STATES_LEVEL, f"{b['state']} to {a['state']}"
    if "country" in allowed and same["country"]:
        return "country", b["country"]
    raise BracketRouteRefusedError(
        f"bracket fixes share no common region at {sorted(allowed)}: before geocodes to "
        f"{before_place!r}, after to {after_place!r}"
    )


def bracket_fixes(path: Path, workdir: Optional[Path] = None):
    """Timed trackpoints of ``path`` (a GPX, or a GoPro recording whose GPMF telemetry
    is extracted to GPX under ``workdir`` first), oldest first."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        out_dir = Path(workdir) if workdir else path.parent / "_bracket_gpx"
        gpx = extract_gopro_gpx(path, out_dir)
        if gpx is None:
            raise BracketRouteRefusedError(f"bracket file {path} has no GPS telemetry")
    elif suffix in _GPX_SUFFIXES:
        gpx = path
    else:
        raise BracketRouteRefusedError(f"bracket file {path} is neither a .gpx nor a GoPro video")
    points = [p for p in parse_gpx_file(Path(gpx)).trackpoints if p.timestamp is not None]
    if not points:
        raise BracketRouteRefusedError(f"bracket file {path} has no timed GPS fix")
    return sorted(points, key=lambda p: p.timestamp)


def _footage_dates(dates: Iterable) -> list:
    out = []
    for d in dates or []:
        text = str(d)[:10]
        try:
            out.append(datetime.strptime(text, "%Y-%m-%d").date())
        except ValueError:
            raise BracketRouteRefusedError(f"footage date {d!r} is not YYYY-MM-DD") from None
    return out


def resolve_bracket_route(
    before_path,
    after_path,
    *,
    geocode_fn: Callable[[float, float], Optional[dict]],
    footage_dates: Iterable = (),
    workdir: Optional[Path] = None,
    max_gap_h_by_level: Optional[dict] = None,
) -> BracketRoute:
    """The bracket route for footage recorded on ``footage_dates`` between the last fix
    of ``before_path`` and the first fix of ``after_path``.

    ``geocode_fn(lat, lon)`` is ``NominatimRoadsClient.reverse_geocode_detailed`` (a dict
    with ``county``/``state``/``country``, or ``None`` on a miss). Refuses (never guesses)
    when: a file has no timed fix; the before fix is not before the after fix; the gap
    exceeds every level's cap in ``max_gap_h_by_level`` (default
    :data:`BRACKET_MAX_GAP_H_BY_LEVEL`; a level whose cap the gap exceeds is never
    claimed); a footage date lies outside the bracket (judged on calendar
    days with one day of slack each side, since footage dates are the camera's local day
    and fix times are UTC); a geocode misses; or there is no common region.
    """
    last = bracket_fixes(before_path, workdir)[-1]
    first = bracket_fixes(after_path, workdir)[0]
    before = BracketFix(float(last.lat), float(last.lon), float(last.timestamp), str(before_path))
    after = BracketFix(float(first.lat), float(first.lon), float(first.timestamp), str(after_path))
    if before.ts >= after.ts:
        raise BracketRouteRefusedError(
            f"bracket BEFORE fix {before.time_iso} ({before.source}) is not before the AFTER "
            f"fix {after.time_iso} ({after.source})"
        )
    gap_h = (after.ts - before.ts) / 3600.0
    caps = dict(max_gap_h_by_level or BRACKET_MAX_GAP_H_BY_LEVEL)
    levels = _levels_within(gap_h, caps)
    if not levels:
        raise BracketRouteRefusedError(
            f"bracket fixes are {gap_h:.1f} h apart (> {max(caps.values()):g} h): "
            f"{before.source} and {after.source} do not bracket one drive"
        )
    lo = datetime.fromtimestamp(before.ts, tz=timezone.utc).date() - timedelta(days=1)
    hi = datetime.fromtimestamp(after.ts, tz=timezone.utc).date() + timedelta(days=1)
    for day in _footage_dates(footage_dates):
        if not lo <= day <= hi:
            raise BracketRouteRefusedError(
                f"footage date {day} lies outside the bracket {before.time_iso} .. {after.time_iso}"
            )
    places = []
    for fix in (before, after):
        place = geocode_fn(fix.lat, fix.lon)
        if not isinstance(place, dict) or not any(_slot(place, k) for k in _PLACE_KEYS):
            raise BracketRouteRefusedError(
                f"bracket fix {fix.lat:.5f},{fix.lon:.5f} ({fix.source}) did not geocode"
            )
        places.append({k: place.get(k) for k in _PLACE_KEYS})
    level, label = common_region(places[0], places[1], levels)
    return BracketRoute(
        before=before,
        after=after,
        gap_h=gap_h,
        before_place=places[0],
        after_place=places[1],
        level=level,
        label=label,
    )


def _levels_within(gap_h: float, caps: dict) -> tuple:
    """The place levels whose gap cap ``gap_h`` is inside."""
    return tuple(k for k in _PLACE_KEYS if gap_h <= float(caps.get(k, 0.0)))


def recomputed_region(record: Optional[dict]) -> Optional[tuple[str, str]]:
    """``(level, label)`` a stored record's own geocoded fixes earn under the CURRENT
    rules (no geocoder, no fix files), or ``None`` for a record that is unreadable or
    earns nothing. A resumed drive compares it with the label the record was burned
    under (``drive_orchestrator.relabel_bracket_on_resume``)."""
    try:
        levels = _levels_within(float(record["gap_h"]), BRACKET_MAX_GAP_H_BY_LEVEL)
        return common_region(record["before_place"], record["after_place"], levels)
    except (KeyError, TypeError, ValueError, AttributeError, BracketRouteRefusedError):
        return None


def bracket_outcome(record: dict) -> dict:
    """An agreeing OST-system-review outcome for a label that is the bracket label:
    the reason names both fixes, their gap and the common region, so the record
    carries the whole basis of the claim."""
    before, after = record.get("before") or {}, record.get("after") or {}
    if record.get("level") == STATES_LEVEL:
        ends = (
            f"geocode to {_slot(record.get('before_place'), 'state')!r} and "
            f"{_slot(record.get('after_place'), 'state')!r}, the two ends of the leg "
            f"{record.get('label')!r} ({STATES_LEVEL} level)"
        )
    else:
        ends = f"both geocode to {record.get('label')!r} ({record.get('level')} level)"
    return {
        "refused": False,
        "label_requested": record.get("label"),
        "reason": (
            f"bracket route: last fix before the footage {before.get('time')} and first fix "
            f"after {after.get('time')} ({float(record.get('gap_h') or 0):.1f} h apart) "
            f"{ends}"
        ),
    }
