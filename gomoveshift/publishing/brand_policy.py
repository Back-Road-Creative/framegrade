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

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import yaml

_CONSOLE_ROOT = Path(__file__).resolve().parent.parent / "console"
_SHIPPED_BRANDS_DIR = _CONSOLE_ROOT / "brands"

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
    path = _SHIPPED_BRANDS_DIR / slug / "brand.yaml"
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
