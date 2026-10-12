"""Tests for playlist_resolver — generator-correctness of playlist assignment.

``resolve_playlist(metadata, config) -> str`` returns the
playlist name to assign for a given LF. The function is total — every
input maps to exactly one playlist via the config-defined rules or the
``default``. No runtime gate; the contract is enforced by construction.
"""

from __future__ import annotations

import pytest

from gomoveshift.config import Config
from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.longform_metadata import HIGHLIGHTS_TRIM_THRESHOLD
from gomoveshift.pipeline.playlist_resolver import resolve_playlist


@pytest.fixture
def cfg_default() -> Config:
    """Config with the canonical youtube_playlists block,
    plus the format-aware highlights rules (must precede the region-only
    rules so a highlight-band LF matches first)."""
    return Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [
                        {
                            "match": {"region": "ireland", "format": "highlights"},
                            "playlist_name": "Ireland Highlights",
                        },
                        {
                            "match": {"region": "usa", "format": "highlights"},
                            "playlist_name": "USA Highlights",
                        },
                        {
                            "match": {"region": "ireland"},
                            "playlist_name": "Ireland Drives",
                        },
                        {
                            "match": {"region": "usa"},
                            "playlist_name": "USA Drives",
                        },
                    ],
                    "default": "All Drives",
                }
            }
        }
    )


def test_resolved_country_routes_ireland(cfg_default: Config):
    """A non-US drive carries its GEOCODED country in the resolved model, so
    a real production label — which ends in the county/province (``", Kerry"``),
    never ``", Ireland"`` — still routes to the Ireland playlist off the model,
    not the label parse (which would yield ``"kerry"`` → All Drives)."""
    metadata = {
        "location": "Killarney to Dingle via N86, Kerry",
        "resolved_location": ResolvedLocation(country_code="Ireland"),
    }
    assert resolve_playlist(metadata, cfg_default) == "Ireland Drives"


def test_country_derived_region_usa(cfg_default: Config):
    metadata = {"location": "Blue Ridge Parkway, North Carolina, USA"}
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_county_state_label_no_country_routes_to_usa(cfg_default: Config):
    """Dominant-county labels (``"County, State"``, no country component)
    must route to the USA playlist via trailing-US-state recognition."""
    metadata = {"location": "Beaufort County, South Carolina"}
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_state_abbr_label_no_country_routes_to_usa(cfg_default: Config):
    metadata = {"location": "Port Royal to Lady's Island, SC"}
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_non_us_two_letter_last_component_not_usa(cfg_default: Config):
    """A 2-letter token that is not a US state abbr must not route to USA."""
    metadata = {"location": "Some Town, IE"}
    assert resolve_playlist(metadata, cfg_default) == "All Drives"


def test_explicit_region_overrides_country(cfg_default: Config):
    """Explicit metadata.region beats the country derivation."""
    metadata = {
        "location": "Blue Ridge Parkway, North Carolina, Ireland",
        "region": "usa",
    }
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_no_matching_rule_returns_default(cfg_default: Config):
    metadata = {"location": "Some Road, Somewhere, Belgium"}
    assert resolve_playlist(metadata, cfg_default) == "All Drives"


def test_empty_rules_returns_default():
    cfg = Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [],
                    "default": "All Drives",
                }
            }
        }
    )
    metadata = {"location": "Anywhere, Ireland"}
    assert resolve_playlist(metadata, cfg) == "All Drives"


def test_series_suffix_tolerance(cfg_default: Config):
    """Trailing ' — Part N' must be stripped before the label-parse fallback
    derives the country (a US county/state label carries the title's suffix)."""
    metadata = {"location": "Beaufort County, South Carolina — Part 1"}
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_resolved_country_case_insensitive(cfg_default: Config):
    """The geocoded country routes case-insensitively (Nominatim casing varies)."""
    metadata = {
        "location": "Killarney to Dingle via N86, Kerry",
        "resolved_location": ResolvedLocation(country_code="IRELAND"),
    }
    assert resolve_playlist(metadata, cfg_default) == "Ireland Drives"


def test_explicit_region_case_insensitive(cfg_default: Config):
    metadata = {"location": "Killarney to Dingle via N86, Kerry", "region": "Ireland"}
    assert resolve_playlist(metadata, cfg_default) == "Ireland Drives"


def test_missing_location_returns_default(cfg_default: Config):
    """No location at all → can't derive country → default."""
    assert resolve_playlist({}, cfg_default) == "All Drives"
    assert resolve_playlist({"location": ""}, cfg_default) == "All Drives"


def test_code_level_default_is_unsorted_fallback():
    """Both zero-config defaults are the fallback title: the uploader
    get-or-creates playlists by exact title, so a stale code default would
    silently create a second playlist on the next config-less run."""
    from gomoveshift.config import YouTubePlaylistsConfig

    assert YouTubePlaylistsConfig().default == "Unsorted (fallback)"
    cfg = Config.model_validate({"drive": {}})
    assert cfg.drive.youtube_playlists.default == "Unsorted (fallback)"


# The playlist resolver consumes the canonical ResolvedLocation.
def test_resolved_location_is_authoritative_over_label(cfg_default: Config):
    """A resolved model's is_us beats a misleading display label."""
    md = {"location": "Somewhere, Ireland", "resolved_location": ResolvedLocation(is_us=True)}
    assert resolve_playlist(md, cfg_default) == "USA Drives"


def test_empty_resolved_falls_back_to_label(cfg_default: Config):
    """A resolved shell with no signal must not mask the label derivation."""
    md = {"location": "Beaufort County, South Carolina", "resolved_location": ResolvedLocation()}
    assert resolve_playlist(md, cfg_default) == "USA Drives"


# Format-aware routing: highlight-band LFs route to the Highlights
# playlists ("Ireland Highlights" / "USA Highlights") instead of the Drives
# playlists.


def test_ireland_highlights_band_routes_to_highlights(cfg_default: Config):
    metadata = {
        "location": "Killarney to Dingle via N86, Kerry",
        "region": "ireland",
        "trim_ratio": HIGHLIGHTS_TRIM_THRESHOLD - 0.1,
    }
    assert resolve_playlist(metadata, cfg_default) == "Ireland Highlights"


def test_ireland_at_threshold_routes_to_drives(cfg_default: Config):
    """Boundary: trim_ratio == threshold is the "full" band (strictly-below
    mirrors longform_metadata._select_verb's ``trim_ratio < THRESHOLD``)."""
    metadata = {
        "location": "Killarney to Dingle via N86, Kerry",
        "region": "ireland",
        "trim_ratio": HIGHLIGHTS_TRIM_THRESHOLD,
    }
    assert resolve_playlist(metadata, cfg_default) == "Ireland Drives"


def test_usa_highlights_band_routes_to_highlights(cfg_default: Config):
    metadata = {
        "location": "Blue Ridge Parkway, North Carolina, USA",
        "trim_ratio": HIGHLIGHTS_TRIM_THRESHOLD - 0.1,
    }
    assert resolve_playlist(metadata, cfg_default) == "USA Highlights"


def test_usa_at_threshold_routes_to_drives(cfg_default: Config):
    metadata = {
        "location": "Blue Ridge Parkway, North Carolina, USA",
        "trim_ratio": HIGHLIGHTS_TRIM_THRESHOLD,
    }
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_no_trim_ratio_signal_unchanged(cfg_default: Config):
    """No ``trim_ratio`` key at all → no ``format`` derived → region-only
    rules still apply exactly as before format-aware routing existed."""
    metadata = {"location": "Blue Ridge Parkway, North Carolina, USA"}
    assert resolve_playlist(metadata, cfg_default) == "USA Drives"


def test_explicit_route_story_format_routes_to_its_own_playlist():
    """``route_story/metadata.py`` stamps an explicit ``format: "route_story"``
    (no ``trim_ratio`` involved). A config rule matching on it alone -- no
    ``region`` key required -- wins over any region-only rule, same override
    mechanism as the explicit-``format`` highlights case above."""
    cfg = Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [
                        {
                            "match": {"format": "route_story"},
                            "playlist_name": "Route Stories",
                        },
                        {
                            "match": {"region": "usa"},
                            "playlist_name": "USA Drives",
                        },
                    ],
                    "default": "All Drives",
                }
            }
        }
    )
    metadata = {
        "location": "Blue Ridge Parkway, North Carolina, USA",
        "format": "route_story",
    }
    assert resolve_playlist(metadata, cfg) == "Route Stories"


# ── Shorts routing ──────────────────────────────────────────────────────────
# Shorts carry a PLACE label in ``sampled_regions`` ("West Cork", "Unicoi,
# Tennessee"), never a country. A place label handed to the resolver as
# ``region`` matches no rule.


def _config_yaml_playlists():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    return Config.model_validate(
        {"drive": {"youtube_playlists": raw["drive"]["youtube_playlists"]}}
    )


@pytest.fixture(params=["code_defaults", "config_yaml"])
def cfg_shipped(request) -> Config:
    if request.param == "code_defaults":
        return Config.model_validate({"drive": {}})
    return _config_yaml_playlists()


def test_shorts_rules_route_to_shorts_playlists(cfg_shipped):
    assert (
        resolve_playlist({"format": "shorts", "region": "ireland"}, cfg_shipped) == "Ireland Shorts"
    )
    assert resolve_playlist({"format": "shorts", "region": "usa"}, cfg_shipped) == "USA Shorts"


def test_shorts_rules_do_not_capture_long_forms(cfg_shipped):
    assert resolve_playlist({"region": "ireland", "trim_ratio": 0.95}, cfg_shipped) == (
        "Ireland Drives"
    )
    assert resolve_playlist({"region": "usa", "trim_ratio": 0.95}, cfg_shipped) == "USA Drives"
    assert resolve_playlist({}, cfg_shipped) == "Unsorted (fallback)"


@pytest.mark.parametrize(
    ("label", "country"),
    [
        ("West Cork", "ireland"),
        ("North Tipperary", "ireland"),
        ("County Kerry", "ireland"),
        ("Kerry", "ireland"),
        ("Cork, Ireland", "ireland"),
        ("County Kerry, Munster", "ireland"),
        ("Cork, Munster", "ireland"),
        ("West Cork, Munster", "ireland"),
        ("Unicoi, Tennessee", "usa"),
        ("Unicoi, TN", "usa"),
        ("Johnson City, Tennessee", "usa"),
        ("Sullivan County, Tennessee", "usa"),
    ],
)
def test_short_region_derives_country_from_place_label(label, country):
    from gomoveshift.pipeline.playlist_resolver import short_region

    assert short_region((label,)) == country


@pytest.mark.parametrize("labels", [(), ("",), ("Springfield",), ("Nowhere Land",)])
def test_short_region_unknown_place_is_empty(labels):
    from gomoveshift.pipeline.playlist_resolver import short_region

    assert short_region(labels) == ""


def test_short_region_uses_first_derivable_label():
    from gomoveshift.pipeline.playlist_resolver import short_region

    assert short_region(("Springfield", "Unicoi, Tennessee")) == "usa"
