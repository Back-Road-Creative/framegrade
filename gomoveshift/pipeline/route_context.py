"""Route context extraction from GPS chapters and location strings.

Parses chapter data (from GPS geocoding) and user-provided location strings
to produce a RouteContext with structured fields for title generation.

The title is derived from GPS data (chapter towns), NOT from the --location
string. The location string is only used for region/country context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A county (or county-equivalent) label in either form: Irish ``"County
# <Name…>"`` or US ``"<Name…> County"``, each tolerating a MULTI-WORD name
# ("Prince William County", "County Kerry"). The trailing US form also covers
# the non-"County" equivalents geocoders return — Louisiana Parish, Alaska
# Borough / Census Area, and Virginia independent City ("Iberia Parish",
# "Matanuska-Susitna Borough", "Nome Census Area", "Richmond City") — each of
# which requires a preceding name, so a BARE "City"/"Borough" never matches.
# Shared by the chapter parser and the gate-7 cross-check (content_gates
# imports this) so the two never drift.
_COUNTY = r"(?:County(?:\s+[\w.'-]+)+|(?:[\w.'-]+\s+)+(?:County|Parish|Borough|Census\s+Area|City))"

# ``Entering <town>, <county>`` — the county may carry a trailing named-route
# suffix (", Slea Head Drive" may be appended), so it is no longer anchored to ``$``.
_ENTERING_RE = re.compile(rf"^Entering\s+(.+?),\s+({_COUNTY})(?:,\s+.+)?$")

# A bare county label (used to keep a county out of the country slot).
_COUNTY_FORM_RE = re.compile(_COUNTY)


@dataclass(frozen=True)
class RouteContext:
    """Structured route information extracted from chapters + location string."""

    location: str
    gps_start_town: str | None = None
    gps_end_town: str | None = None
    region: str | None = None
    county: str | None = None
    country: str | None = None
    gps_towns: list[str] = field(default_factory=list)

    @property
    def smart_start(self) -> str | None:
        """First non-obscure, non-business GPS town."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        for town in self.gps_towns:
            if _is_obscure(town):
                continue
            if not is_valid_chapter_placename(town):
                continue
            return town
        # Every GPS town was rejected — fall back to None (not the raw first
        # town we just filtered out) so make_default_title uses its
        # location-string branch instead of burning a rejected name.
        return None

    @property
    def smart_end(self) -> str | None:
        """Last non-obscure, non-business GPS town."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        for town in reversed(self.gps_towns):
            if _is_obscure(town):
                continue
            if not is_valid_chapter_placename(town):
                continue
            return town
        # Every GPS town was rejected — fall back to None (not the raw last
        # town we just filtered out); see smart_start.
        return None

    def make_default_title(self) -> str:
        """Build title from location string or GPS towns + region/country.

        Prefers the --location string when it contains a named geographic
        feature (Peninsula, Pass, Valley, Coast, etc.) since those are more
        recognizable than obscure townland names from GPS geocoding.

        Ensures title is 40-70 chars for optimal YouTube SEO.
        """
        # If location contains a well-known geographic feature, prefer it
        _geo_features = (
            "Peninsula",
            "Pass",
            "Valley",
            "Coast",
            "Mountains",
            "Hills",
            "Bay",
            "Head",
            "Island",
            "Lake",
            "Ring",
        )
        loc_primary = self.location.split(",")[0].strip() if self.location else ""
        # Word-boundary match: substring matching wrongly
        # forced the feature title form on towns that merely CONTAIN a feature
        # word ("Pass" in "Passage East", "Head" in "Headford").
        has_geo_feature = any(re.search(rf"\b{re.escape(f)}\b", loc_primary) for f in _geo_features)

        if has_geo_feature:
            short = _shorten_location(self.location)
            country = self.country or ""
            if country and country not in short:
                base = f"{short} | {country}"
            else:
                base = short
        elif self.smart_start and self.smart_end and self.smart_start != self.smart_end:
            start = self.smart_start
            end = self.smart_end
            region_kw = _region_keyword(self.region) if self.region else None
            parts = [f"{start} to {end}"]
            suffix_parts = []
            if region_kw:
                suffix_parts.append(region_kw)
            if self.country:
                suffix_parts.append(self.country)
            if suffix_parts:
                parts.append(", ".join(suffix_parts))
            base = " | ".join(parts)
        else:
            # Single town or no GPS — fallback to location string
            short = _shorten_location(self.location)
            country = self.country or ""
            if country and country not in short:
                base = f"{short} | {country}"
            else:
                base = short

        # Pad short titles with a factual format token for YouTube SEO (target
        # 40-70 chars). No genre word ("Scenic Drive"): "not all footage is
        # scenic drives", so the pad stays route/format-only.
        if len(base) < 40:
            base = f"POV Driving {base}"

        return base[:100]


def _is_obscure(town: str) -> bool:
    """Single word >12 chars is likely an obscure townland name."""
    return " " not in town and len(town) > 12


def _region_keyword(region: str) -> str:
    """Extract short keyword from region: 'Beara Peninsula' -> 'Beara'."""
    for suffix in ("Peninsula", "Valley", "Coast", "Mountains", "Hills"):
        if region.endswith(f" {suffix}") and len(region.split()) > 1:
            return region.replace(f" {suffix}", "").strip()
    return region


def _shorten_location(location: str) -> str:
    """Return primary segment of a comma-separated location string."""
    if not location or not location.strip():
        return ""
    parts = [p.strip() for p in location.split(",")]
    primary = parts[0]
    if len(parts) >= 3 and len(primary) < 10:
        second = parts[1]
        if not second.startswith("County"):
            return f"{primary}, {second}"
    return primary


def extract_route_context(location: str, chapters: list[dict]) -> RouteContext:
    """Factory: parse chapters + location string -> RouteContext."""
    # --- Parse chapters for GPS towns ---
    gps_towns: list[str] = []
    county: str | None = None

    for ch in chapters:
        title = ch.get("title", "")
        m = _ENTERING_RE.match(title)
        if m:
            town = m.group(1)
            gps_towns.append(town)
            if county is None:
                county = m.group(2)

    gps_start_town = gps_towns[0] if gps_towns else None
    gps_end_town = gps_towns[-1] if gps_towns else None

    # --- Parse location string for region/country context only ---
    parts = [p.strip() for p in location.split(",")]
    country = parts[-1] if len(parts) >= 2 else None
    # A county label ("County Kerry" / "Prince William County") is NOT a country
    # — leaving it here stamps a spurious "| County Kerry" country noun on the
    # title. Drop it so the country slot stays empty for county-only locations.
    if country is not None and _COUNTY_FORM_RE.fullmatch(country):
        country = None
    if county is None:
        for p in parts:
            if p.startswith("County"):
                county = p
                break

    # Extract region from location string (first segment before "to" or
    # second segment for "and Return" patterns)
    primary = parts[0] if parts else ""
    region: str | None = None

    if "and Return" in primary:
        if len(parts) >= 2 and not parts[1].startswith("County"):
            region = parts[1]
    elif " to " in primary:
        before_to = primary.split(" to ", 1)[0].strip()
        region = before_to
    else:
        # Try non-county, non-country segments as region
        for p in parts:
            if not p.startswith("County") and p != country and p != primary:
                region = p
                break

    return RouteContext(
        location=location,
        gps_start_town=gps_start_town,
        gps_end_town=gps_end_town,
        region=region,
        county=county,
        country=country,
        gps_towns=gps_towns,
    )
