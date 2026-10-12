"""User-final-say OST override loader.

When the automated OST derivation can't produce text that satisfies the
human reviewer (small local place names, unindexed towns, regional nicknames
like "Lowcountry"), the operator can drop an ``ost_overrides.json`` file
next to the pipeline state and pin exact text for specific episodes.

File schema:
    {
      "episode_0": "Frogmore to Port Royal via Overland Trail",
      "episode_3": "Port Royal to Gardens Corner via Trask Parkway",
      "drive": "Lowcountry to Blue Ridge"
    }

Keys:
  * ``episode_<N>`` — 0-indexed episode override. Out-of-range indices are
    ignored with a warning.
  * ``drive`` — drive-level (super LF) override.

Any other top-level key is ignored with a warning. Non-string values are
ignored. Missing file → empty dict. Malformed JSON → empty dict + warning.

Applied AFTER derivation. Override values are revalidated with
``validate_ost`` before use (so a gate cannot be bypassed): a
typo'd/placeholder/garbage override is ignored with a warning and the
derived value keeps flowing — mirroring the per-LF override revalidation
in ``gps_enrichment.validate_burned_ost_text`` — so bad override text can
never burn verbatim on the episode path.
"""

import json
import logging
import re
from pathlib import Path

from .ost_validator import validate_ost

logger = logging.getLogger(__name__)

_OVERRIDE_FILENAME = "ost_overrides.json"
_EPISODE_KEY_RE = re.compile(r"^episode_(\d+)$")


def load_overrides(output_dir) -> dict:
    """Load ost_overrides.json from ``output_dir``. Returns ``{}`` on any
    failure (missing, malformed, wrong type). Warnings are logged.
    """
    override_path = Path(output_dir) / _OVERRIDE_FILENAME
    if not override_path.exists():
        return {}

    try:
        raw = override_path.read_text()
    except OSError as e:
        logger.warning("  Could not read OST override file %s: %s", override_path, e)
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(
            "  OST override file %s is malformed, ignoring: %s",
            override_path,
            e,
        )
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "  OST override file %s top-level is not an object, ignoring",
            override_path,
        )
        return {}

    return data


def apply_overrides(
    episode_locations: list[str],
    drive_location: str,
    overrides: dict,
) -> tuple[list[str], str]:
    """Return new (episode_locations, drive_location) with overrides applied.

    Unknown keys, out-of-range episode indices, and non-string values are
    ignored with warnings. The original lists are not mutated.
    """
    if not overrides:
        return list(episode_locations), drive_location

    new_episodes = list(episode_locations)
    new_drive = drive_location

    for key, value in overrides.items():
        if not isinstance(value, str):
            logger.warning(
                "  OST override %r has non-string value %r, ignoring",
                key,
                value,
            )
            continue

        # Revalidate before use — episode/drive overrides previously shipped
        # verbatim while per-LF overrides revalidated (gps_enrichment).
        ok, reason = validate_ost(value, "lf")
        if not ok:
            logger.warning(
                "  OST override %r value %r fails validation (%s), ignoring",
                key,
                value,
                reason,
            )
            continue

        if key == "drive":
            new_drive = value
            logger.info("  OST override applied: drive → %r", value)
            continue

        match = _EPISODE_KEY_RE.match(key)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(new_episodes):
                new_episodes[idx] = value
                logger.info("  OST override applied: episode_%d → %r", idx, value)
            else:
                logger.warning(
                    "  OST override %r is out of range (have %d episodes), ignoring",
                    key,
                    len(new_episodes),
                )
            continue

        logger.warning("  OST override has unknown key %r, ignoring", key)

    return new_episodes, new_drive
