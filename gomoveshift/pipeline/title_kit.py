"""The title / chapter / brand inputs the day render takes from the publish side.

The media engine (``day_pipeline``, ``drive_phases/phase_bc_day_first``) names
titles, builds chapters and reads the tenant's brand policy, but those live in
publishing code the engine must not import (``tests/test_media_boundary.py``). The GMS
composition root builds a :class:`TitleKit` (``publishing/title_kit.py``) and
threads it ``PhaseBCDayFirst`` -> ``run_all_days`` -> ``RunSpec`` ->
``run_day_pipeline``; the engine only calls what the kit holds. A run with no kit
fails loudly at the first step that needs one (:func:`require_title_kit`), the
same contract as ``PipelineContext.require_metadata_generator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class TitleKit:
    """Callables (and one constant) the engine calls instead of importing publish code."""

    # ── day render (``day_pipeline``) ────────────────────────────────
    media_profile: Callable[[], Any]  # () -> MediaProfile of the current brand
    per_location_cap: Callable[[], Optional[int]]  # () -> brand policy per-location cap
    superlong_title: Callable[[str, int, int], str]  # (location, part_index, part_count)
    chapter_generator: Callable[[Any], Any]  # (config) -> ChapterGenerator
    superlong_max_chapters: Callable[[float], int]
    min_chapters: int  # YouTube's minimum chapter count
    day_master_journey: Callable[[Any], Any]  # (day_gps_points) -> journey
    town_events: Callable[[Optional[list]], list]  # (chapters) -> [(label, t_sec)]
    # ── Phase B/C (``phase_bc_day_first``) ───────────────────────────
    lf_title: Callable[..., str]  # build_lf_title
    unique_titles: Callable[..., dict]  # ensure_unique_titles
    published_titles: Callable[[], set]  # load_published_titles
    refresh_title_ledger: Callable[[], dict]  # refresh_channel_title_ledger


def require_title_kit(kit: Optional[TitleKit]) -> TitleKit:
    """Return ``kit``, or raise: the engine never imports the publish-side helpers."""
    if kit is None:
        raise RuntimeError(
            "no title kit was injected: build one with "
            "gomoveshift.publishing.title_kit.build_title_kit() and pass it to "
            "PhaseBCDayFirst(config, title_kit=...) / run_day_pipeline(..., kit=...)"
        )
    return kit
