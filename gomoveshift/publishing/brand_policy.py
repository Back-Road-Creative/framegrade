"""Per-brand policy block in ``console/brands/<slug>/brand.yaml``.

A brand's ``brand.yaml`` may carry an optional ``policy:`` block alongside
the existing ``disabled_tabs:`` key (``console/settings.py``). A missing
file, missing block, or missing key falls back to today's behaviour, so
every brand other than the one that opts in is unaffected. An unknown key
under ``policy:`` raises -- a typo must not silently do nothing.

Keys: ``per_location_cap``, ``weekly_ceiling``, ``daily_ladder``,
``stagger_hours``, ``region_weight``, ``blackout`` (read by
``dispatch._select_due``) and ``quiet_after_alert_hours`` (read by
``dispatch._exclude_quiet_window``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import yaml

_CONSOLE_ROOT = Path(__file__).resolve().parent.parent / "console"
_SHIPPED_BRANDS_DIR = _CONSOLE_ROOT / "brands"


def host_brands_dir() -> Optional[Path]:
    """``GMS_BRANDS_DIR`` (host brand templates), or ``None`` when unset/blank."""
    raw = os.environ.get("GMS_BRANDS_DIR")
    return Path(raw.strip()) if raw and raw.strip() else None


def brand_dir(slug: str, shipped: Optional[Path] = None) -> Path:
    """The directory holding ``slug``'s ``brand.yaml`` / ``settings.yaml``.

    ``<GMS_BRANDS_DIR>/<slug>`` when that variable is set and the directory
    exists, else ``<shipped>/<slug>`` (default: the package's own brands
    dir). A whole-slug overlay: one directory is used for every file of the
    brand, never a per-file merge.
    """
    host = host_brands_dir()
    if host is not None and (host / slug).is_dir():
        return host / slug
    return (shipped if shipped is not None else _SHIPPED_BRANDS_DIR) / slug


_KNOWN_POLICY_KEYS = frozenset(
    {
        "per_location_cap",
        "weekly_ceiling",
        "daily_ladder",
        "stagger_hours",
        "region_weight",
        "blackout",
        "quiet_after_alert_hours",
    }
)


@dataclass(frozen=True)
class BrandPolicy:
    """A brand's ``policy:`` block, defaults matching today's behaviour."""

    #: Max shorts per location/LF. ``None`` means no cap (today's default).
    per_location_cap: Optional[int] = None
    #: Max shorts per platform per rolling week. ``None`` means no cap.
    weekly_ceiling: Optional[int] = None
    #: Daily shorts ladder (e.g. ``(1, 2, 3)``), strictly increasing.
    #: ``None`` means no ladder override.
    daily_ladder: Optional[Tuple[int, ...]] = None
    #: Hours a non-YouTube platform post must wait after the YouTube post.
    #: ``None`` means no stagger.
    stagger_hours: Optional[int] = None
    #: Region name (lowercase) -> relative weight (> 0), as an immutable
    #: mapping. ``None`` means no region weighting.
    region_weight: Optional[Mapping[str, float]] = None
    #: Calendar dates (local to the row's schedule) on which no delivery is
    #: selected. Empty tuple means no blackout (today's default).
    blackout: Tuple[date, ...] = field(default_factory=tuple)
    #: Hours to hold delivery after a suppression alert. ``None`` means no
    #: quiet period.
    quiet_after_alert_hours: Optional[int] = None


def _load_raw_policy_block(slug: str) -> dict:
    path = brand_dir(slug) / "brand.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("policy") or {}


def _validate_int_at_least(block: dict, key: str, minimum: int) -> Optional[int]:
    value = block.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"policy.{key} must be an int >= {minimum} (or omitted), got {value!r}")
    return value


def _validate_daily_ladder(block: dict) -> Optional[Tuple[int, ...]]:
    value = block.get("daily_ladder")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "policy.daily_ladder must be a non-empty sequence of ints >= 1, "
            f"strictly increasing (or omitted), got {value!r}"
        )
    ladder = []
    previous = None
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(
                "policy.daily_ladder must be a non-empty sequence of ints >= 1, "
                f"strictly increasing (or omitted), got {value!r}"
            )
        if previous is not None and item <= previous:
            raise ValueError(
                f"policy.daily_ladder must be strictly increasing (or omitted), got {value!r}"
            )
        ladder.append(item)
        previous = item
    return tuple(ladder)


def _validate_region_weight(block: dict) -> Optional[Mapping[str, float]]:
    value = block.get("region_weight")
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "policy.region_weight must be a non-empty mapping of region -> "
            f"weight > 0 (or omitted), got {value!r}"
        )
    weights = {}
    for region, weight in value.items():
        if not isinstance(region, str) or region == "" or region != region.lower():
            raise ValueError(
                f"policy.region_weight keys must be non-empty lowercase strings, got {region!r}"
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(
                f"policy.region_weight values must be a number > 0, got {weight!r} for {region!r}"
            )
        weights[region] = weight
    return MappingProxyType(weights)


def _validate_blackout(block: dict) -> Tuple[date, ...]:
    value = block.get("blackout")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"policy.blackout must be a sequence of ISO dates (or omitted), got {value!r}"
        )
    dates = []
    for item in value:
        if isinstance(item, date):
            dates.append(item)
            continue
        if isinstance(item, str):
            try:
                dates.append(date.fromisoformat(item))
                continue
            except ValueError:
                pass
        raise ValueError(
            f"policy.blackout must be a sequence of ISO dates (or omitted), got {value!r}"
        )
    return tuple(dates)


def load_brand_policy(slug: str) -> BrandPolicy:
    """Read the optional ``policy:`` block from ``brand.yaml`` for ``slug``.

    A missing file/block/key falls back to :class:`BrandPolicy`'s defaults.
    A present-but-invalid value raises ``ValueError`` naming the offending
    ``policy.<key>``. An unknown key raises ``ValueError`` naming it.
    """
    block = _load_raw_policy_block(slug)
    unknown = set(block) - _KNOWN_POLICY_KEYS
    if unknown:
        bad_key = sorted(unknown)[0]
        raise ValueError(f"policy.{bad_key} is not a recognized policy key")

    cap = _validate_int_at_least(block, "per_location_cap", minimum=1)
    weekly_ceiling = _validate_int_at_least(block, "weekly_ceiling", minimum=1)
    stagger_hours = _validate_int_at_least(block, "stagger_hours", minimum=0)
    daily_ladder = _validate_daily_ladder(block)
    region_weight = _validate_region_weight(block)
    blackout = _validate_blackout(block)
    quiet_after_alert_hours = _validate_int_at_least(block, "quiet_after_alert_hours", minimum=0)

    return BrandPolicy(
        per_location_cap=cap,
        weekly_ceiling=weekly_ceiling,
        daily_ladder=daily_ladder,
        stagger_hours=stagger_hours,
        region_weight=region_weight,
        blackout=blackout,
        quiet_after_alert_hours=quiet_after_alert_hours,
    )


# --- Per-tenant media profile (``media:`` block) ---------------------------
# Defaults reproduce the pre-profile engine literals. The canonical strings
# also live in ``pipeline/ffmpeg_filters.py`` (BRAND_TEXT / WEBSITE_TEXT /
# FONT_PATH); a test pins the two together. Path existence (lut, logo, font)
# is checked at the use site, not at load.

_KNOWN_MEDIA_KEYS = frozenset(
    {
        "lut",
        "logo",
        "font",
        "still_card_style",
        "preset_defaults",
        "upscale",
        "claim_native_resolution_only",
        "brand_text",
        "website_text",
    }
)
_UPSCALE_MODES = frozenset({"resample", "ml", "off"})


@dataclass(frozen=True)
class MediaProfile:
    """A brand's ``media:`` block, defaults matching today's engine behaviour."""

    lut: Optional[str] = None
    logo: Optional[str] = None
    font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    still_card_style: str = "letterbox"
    preset_defaults: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    upscale: str = "off"
    claim_native_resolution_only: bool = True
    brand_text: str = "GoMoveShift"
    website_text: str = "gomoveshift.com"


def _load_raw_media_block(slug: str) -> dict:
    path = brand_dir(slug) / "brand.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("media") or {}


def load_media_profile(slug: str) -> MediaProfile:
    """Read the optional ``media:`` block from ``brand.yaml`` for ``slug``.

    Missing file/block/key falls back to :class:`MediaProfile` defaults. An
    unknown key or an invalid value raises ``ValueError`` naming ``media.<key>``.
    """
    block = _load_raw_media_block(slug)
    unknown = set(block) - _KNOWN_MEDIA_KEYS
    if unknown:
        raise ValueError(f"media.{sorted(unknown)[0]} is not a recognized media key")

    kwargs: dict = {}
    for key in ("lut", "logo"):
        if block.get(key) is not None:
            if not isinstance(block[key], str) or not block[key]:
                raise ValueError(f"media.{key} must be a non-empty string path, got {block[key]!r}")
            kwargs[key] = block[key]
    for key in ("font", "still_card_style", "brand_text", "website_text"):
        if key in block:
            if not isinstance(block[key], str) or not block[key]:
                raise ValueError(f"media.{key} must be a non-empty string, got {block[key]!r}")
            kwargs[key] = block[key]
    if "upscale" in block:
        if block["upscale"] not in _UPSCALE_MODES:
            raise ValueError(
                f"media.upscale must be one of {sorted(_UPSCALE_MODES)}, got {block['upscale']!r}"
            )
        kwargs["upscale"] = block["upscale"]
    if "claim_native_resolution_only" in block:
        if not isinstance(block["claim_native_resolution_only"], bool):
            raise ValueError(
                "media.claim_native_resolution_only must be a bool, "
                f"got {block['claim_native_resolution_only']!r}"
            )
        kwargs["claim_native_resolution_only"] = block["claim_native_resolution_only"]
    if block.get("preset_defaults") is not None:
        if not isinstance(block["preset_defaults"], dict):
            raise ValueError(
                f"media.preset_defaults must be a mapping, got {block['preset_defaults']!r}"
            )
        kwargs["preset_defaults"] = MappingProxyType(dict(block["preset_defaults"]))
    return MediaProfile(**kwargs)
