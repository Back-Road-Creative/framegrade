"""Route Story upload metadata builder.

Turns a ``film_receipt`` dict (``film.film_receipt``) into the
title/description/tags an upload needs, plus a ``format``/``playlist_hint``
a caller writes onto upload metadata for
``pipeline.playlist_resolver.resolve_playlist`` to match a ``route_story``
rule against (an explicit ``metadata["format"]`` wins over the trim-derived
value there -- see ``playlist_resolver.resolve_playlist``).

``film.py``'s ``--credit``/``--metadata`` flags call this builder and write
the metadata JSON beside the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .. import longform_metadata as _longform_metadata

# Same arrow separator as the opening card's derived title
# (``hud.resolve_opening_title``: ``f"{first} → {last}"``).
_ROUTE_NAME_ARROW = "→"

# Editorial defaults for "what kind of trip was this" from the route's
# moving-average speed. Adjustable: these are display-copy bands, not a
# physical constant, chosen to separate on-foot / pedal / motorized trips at
# typical paces. A route with no known moving-average speed (rare -- only a
# fully elevation-less, speed-less GPS source) reads as "Driving", matching
# ``hud.motion_cell_mode``'s "unknown reads speed" default.
HIKING_MAX_SPEED_MPS = 3.5
RIDING_MAX_SPEED_MPS = 9.0

METERS_PER_MILE = 1609.344
METERS_PER_FOOT = 0.3048

# YouTube's tag budget: <=500 total characters across all tags, and this
# builder additionally caps at 15 tags / 30 chars each so a route with many
# map labels never produces an unreadable tag wall.
MAX_TAGS = 15
MAX_TAG_CHARS = 30
MAX_TOTAL_TAG_CHARS = 500


class DuplicateTitleError(Exception):
    """Raised when a resolved title already appears in the published-titles ledger."""


class LedgerUnreadableError(Exception):
    """Raised when the published-titles ledger status is UNREADABLE.

    Title uniqueness can't be verified in that state, so a hard-gating
    caller must fail rather than silently treat it as "no known titles".
    """


@dataclass(frozen=True)
class RouteStoryMetadata:
    """Upload metadata derived from a route-story film receipt."""

    title: str
    description: str
    tags: tuple[str, ...]
    recording_date: str
    format: str = field(default="route_story")
    playlist_hint: str = field(default="route_story")


def _activity_for_speed(moving_average_speed_mps: float | None) -> str:
    if moving_average_speed_mps is None:
        return "Driving"
    if moving_average_speed_mps <= HIKING_MAX_SPEED_MPS:
        return "Hiking"
    if moving_average_speed_mps <= RIDING_MAX_SPEED_MPS:
        return "Riding"
    return "Driving"


def _derived_title(activity: str, names: Sequence[str], recorded_on: str) -> str:
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    if len(deduped) >= 2:
        route_name = f"{deduped[0]} {_ROUTE_NAME_ARROW} {deduped[-1]}"
    elif deduped:
        route_name = deduped[0]
    else:
        route_name = recorded_on
    return f"{activity} {route_name}"


def _resolve_title(
    *,
    activity: str | None,
    moving_average_speed_mps: float | None,
    title: str | None,
    names: Sequence[str],
    recorded_on: str,
) -> str:
    """The single title-derivation path shared by :func:`build_route_story_metadata`
    and :func:`preflight_title` so the two can never drift.

    An explicit ``activity`` (a resolved ``ActivityProfile.name``) wins;
    otherwise the activity is guessed from ``moving_average_speed_mps`` via
    :func:`_activity_for_speed` (``None`` reads as "unknown speed", i.e.
    "Driving" -- the same default a genuinely speed-less route gets).
    """
    resolved_activity = (
        activity.capitalize() if activity else _activity_for_speed(moving_average_speed_mps)
    )
    resolved_title = title.strip() if title and title.strip() else None
    if resolved_title is None:
        resolved_title = _derived_title(resolved_activity, names, recorded_on)
    return resolved_title


def preflight_title(
    *,
    recorded_on: str,
    title: str | None,
    activity: str | None,
    names: Sequence[str],
) -> str:
    """Resolve the eventual title and gate it against the ledger before any render runs.

    Uses :func:`_resolve_title`, the exact derivation :func:`build_route_story_metadata`
    uses, so a preflight pass and the post-render metadata can never disagree
    given the same inputs. The one input this early check cannot have is the
    route's real moving-average speed (only known after the render measures
    the qualified track) -- when ``activity`` is ``None`` (auto-guessed
    activity) this falls back to the same "unknown speed" default as
    :func:`_activity_for_speed`, so an auto-guessed activity's preflight
    title may occasionally differ from the post-render title. A caller
    relying on this for a hard gate should also keep the post-render
    :func:`check_title_against_ledger` as the fresh, authoritative check
    right before anything is written.

    Raises :class:`LedgerUnreadableError`/:class:`DuplicateTitleError`, same
    as :func:`check_title_against_ledger`.
    """
    resolved_title = _resolve_title(
        activity=activity,
        moving_average_speed_mps=None,
        title=title,
        names=names,
        recorded_on=recorded_on,
    )
    published_titles, status = _longform_metadata.read_published_titles()
    if status == _longform_metadata.LEDGER_UNREADABLE:
        raise LedgerUnreadableError(
            "published-titles ledger is unreadable; title uniqueness can't be verified"
        )
    if resolved_title in published_titles:
        raise DuplicateTitleError(f"title already published: {resolved_title!r}")
    return resolved_title


def _format_distance(distance_m: float) -> str:
    miles = distance_m / METERS_PER_MILE
    km = distance_m / 1000.0
    return f"{miles:.1f} mi ({km:.1f} km)"


def _format_ascent(ascent_m: float) -> str:
    feet = ascent_m / METERS_PER_FOOT
    return f"{round(feet)} ft ({round(ascent_m)} m) climbed"


def _format_moving_time(moving_seconds: float) -> str:
    total_minutes = round(moving_seconds / 60.0)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {minutes}m moving"
    return f"{minutes}m moving"


def _description_lines(
    route_summary: Mapping[str, object],
    recorded_on: str,
    credit_lines: Sequence[str],
) -> list[str]:
    lines: list[str] = []
    distance_m = route_summary.get("distance_m")
    if isinstance(distance_m, (int, float)):
        lines.append(_format_distance(float(distance_m)))
    ascent_m = route_summary.get("ascent_m")
    if isinstance(ascent_m, (int, float)):
        lines.append(_format_ascent(float(ascent_m)))
    moving_seconds = route_summary.get("moving_seconds")
    if isinstance(moving_seconds, (int, float)):
        lines.append(_format_moving_time(float(moving_seconds)))
    lines.append(f"Recorded {recorded_on}")
    if credit_lines:
        lines.append("")
        lines.extend(credit_lines)
    return lines


def _credit_lines(receipt: Mapping[str, object], credits: Sequence[str]) -> list[str]:
    """Imagery credits in national -> regional -> local order, then extras.

    Mirrors the on-screen credit order (national basemap first, most local
    last) so the description and the burned-in credits never disagree.
    """
    lines: list[str] = []
    national = receipt.get("national_imagery")
    if isinstance(national, Mapping) and national.get("credit"):
        lines.append(f"Imagery: {national['credit']}")
    regional = receipt.get("regional_imagery")
    if isinstance(regional, Sequence):
        for raster in regional:
            if isinstance(raster, Mapping) and raster.get("credit"):
                lines.append(f"Imagery: {raster['credit']}")
    local = receipt.get("local_imagery")
    if isinstance(local, Mapping) and local.get("credit"):
        lines.append(f"Imagery: {local['credit']}")
    for extra in credits:
        if extra:
            lines.append(f"Imagery: {extra}")
    return lines


def _tags(activity: str, names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    tags: list[str] = []

    def _add(candidate: str) -> None:
        trimmed = candidate.strip()[:MAX_TAG_CHARS]
        key = trimmed.lower()
        if not trimmed or key in seen:
            return
        seen.add(key)
        tags.append(trimmed)

    _add(activity)
    _add("route story")
    for name in names:
        _add(name)

    budget_ok: list[str] = []
    total_chars = 0
    for tag in tags[:MAX_TAGS]:
        if total_chars + len(tag) > MAX_TOTAL_TAG_CHARS:
            break
        budget_ok.append(tag)
        total_chars += len(tag)
    return tuple(budget_ok)


def build_route_story_metadata(
    receipt: Mapping[str, object],
    *,
    recorded_on: str,
    title: str | None = None,
    credits: Sequence[str] = (),
    published_titles: set[str] | None = None,
    ledger_status: str | None = None,
    activity: str | None = None,
) -> RouteStoryMetadata:
    """Build upload metadata from a route-story ``film_receipt`` dict.

    Title: the explicit ``title`` verbatim (stripped), else
    ``"<Activity> <route name>"`` -- ``Activity`` from the explicit
    ``activity`` (a resolved ``route_story.activity.ActivityProfile.name``,
    e.g. ``"cycling"``) when given, else guessed from the route's moving
    average speed (see ``_activity_for_speed``) -- and the route name from
    ``receipt["map_labels"]["names"]`` first/last, mirroring how the film's
    own opening card names the trip when no ``--title`` is given
    (``hud.resolve_opening_title`` -- nearest label at
    the track's start and end, joined with the same arrow separator; falls
    back to the recording date when no label is close enough).

    Raises :class:`LedgerUnreadableError` when ``ledger_status`` is
    ``LEDGER_UNREADABLE`` (uniqueness can't be verified) and
    :class:`DuplicateTitleError` when the resolved title is already in
    ``published_titles``.
    """
    route_summary = receipt.get("route_summary")
    if not isinstance(route_summary, Mapping):
        route_summary = {}
    map_labels = receipt.get("map_labels")
    names: Sequence[str] = ()
    if isinstance(map_labels, Mapping):
        raw_names = map_labels.get("names")
        if isinstance(raw_names, Sequence):
            names = [str(name) for name in raw_names]

    moving_average_speed_mps = route_summary.get("moving_average_speed_mps")
    resolved_title = _resolve_title(
        activity=activity,
        moving_average_speed_mps=(
            moving_average_speed_mps if isinstance(moving_average_speed_mps, (int, float)) else None
        ),
        title=title,
        names=names,
        recorded_on=recorded_on,
    )
    resolved_activity = (
        activity.capitalize()
        if activity
        else _activity_for_speed(
            moving_average_speed_mps if isinstance(moving_average_speed_mps, (int, float)) else None
        )
    )

    if ledger_status == _longform_metadata.LEDGER_UNREADABLE:
        raise LedgerUnreadableError(
            "published-titles ledger is unreadable; title uniqueness can't be verified"
        )
    if published_titles is not None and resolved_title in published_titles:
        raise DuplicateTitleError(f"title already published: {resolved_title!r}")

    credit_lines = _credit_lines(receipt, credits)
    description = "\n".join(_description_lines(route_summary, recorded_on, credit_lines))
    tags = _tags(resolved_activity, names)

    return RouteStoryMetadata(
        title=resolved_title,
        description=description,
        tags=tags,
        recording_date=recorded_on,
    )


def check_title_against_ledger(meta: RouteStoryMetadata) -> None:
    """Re-check ``meta.title`` against the live published-titles ledger.

    Raises the same errors as :func:`build_route_story_metadata` when the
    ledger can't be verified or the title is already published. Intended
    for a caller that built metadata earlier and wants a fresh check right
    before an upload commits.
    """
    published_titles, status = _longform_metadata.read_published_titles()
    if status == _longform_metadata.LEDGER_UNREADABLE:
        raise LedgerUnreadableError(
            "published-titles ledger is unreadable; title uniqueness can't be verified"
        )
    if meta.title in published_titles:
        raise DuplicateTitleError(f"title already published: {meta.title!r}")
