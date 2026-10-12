"""The one place the pipeline learns how many shorts to plan per long-form.

The number is the SEO manager's decision (``seo_manager`` ``cadence_decisions`` log);
GMS reads it and never computes, defaults or hand-sets a count. An empty store
yields the policy floor seed (``source="seed"``); an unreadable store refuses
and plans nothing. ``--shorts-target`` can only lower the decided value.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from seo_manager.analytics import cadence_decisions

from ..config import get_config

logger = logging.getLogger(__name__)

KNOB = "shorts_per_lf"
_DB_NAME = "cadence_decisions.sqlite3"


class VolumeSourceUnreadableError(RuntimeError):
    """The decision store cannot be read; no shorts are planned."""


@dataclass(frozen=True)
class ShortsDecision:
    value: int
    source: str
    reason_code: str


def shorts_per_lf(db: Optional[Path] = None) -> ShortsDecision:
    """The SEO manager's current shorts-per-LF decision (``source="seed"`` when unset)."""
    if db is None:
        db = Path(get_config().seo.storage_path) / _DB_NAME
    try:
        stored = cadence_decisions.current(db, KNOB)
    except cadence_decisions.DecisionStoreUnreadable as exc:
        logger.error("VOLUME_SOURCE_UNREADABLE reason=%s planned=0", exc)
        raise VolumeSourceUnreadableError(
            f"VOLUME_SOURCE_UNREADABLE reason={exc} planned=0"
        ) from exc
    return ShortsDecision(int(stored.value), stored.source, stored.reason_code)


def apply_flag(decided: int, flag: Optional[int]) -> int:
    """``--shorts-target`` may lower the decided value, never raise it."""
    if flag is not None and flag < decided:
        logger.info("VOLUME_OVERRIDE_LOWER decided=%d flag=%d", decided, flag)
        return flag
    return decided
