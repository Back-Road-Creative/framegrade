"""Meta caption evidence: how each caption opener's posts did on Facebook and Instagram.

For every tracked short with a Facebook or Instagram id, the MEASURED observation
nearest age 7 days (within 12h, the ``evidence_export`` snapshot rule) supplies
one value, and the plan sidecar's ``caption_templates`` names the opener template
that post's caption opened with. Per platform and opener the values become one
Waypost ``CaptionObservation``: their median, their count, and the span of the
snapshots' fetch times.

* Facebook counts Reel **initial plays** (``reel_initial_plays``): Graph answers
  no per-Reel reach for the live Page (``docs/facebook-metrics.md``).
* Instagram counts **reach**, which the collector stores as ``impressions``
  (``InstagramCollector.collect_short_metrics``).

A post with no snapshot, no recorded opener or no readable sidecar is not
evidence. Nothing here ranks: ``seo_manager.analytics.caption_ranking`` decides
what the numbers mean. The nightly ``metrics collect`` writes the result to
``<seo storage>/caption_evidence.json``; the caption generator reads it.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from seo_manager.analytics.caption_ranking import (
    FACEBOOK_REEL_INITIAL_PLAYS_DAY7,
    INSTAGRAM_REEL_REACH_DAY7,
    CaptionEvidence,
    CaptionObservation,
)

from .evidence_export import SNAPSHOT_TARGET, SNAPSHOT_TOLERANCE
from .observation import ObservationStatus

logger = logging.getLogger(__name__)

EVIDENCE_FILE = "caption_evidence.json"

#: platform -> (stored counter, Waypost metric source)
METRICS: Dict[str, Tuple[str, str]] = {
    "facebook": ("reel_initial_plays", FACEBOOK_REEL_INITIAL_PLAYS_DAY7),
    "instagram": ("impressions", INSTAGRAM_REEL_REACH_DAY7),
}

#: project -> ``{short_index: {platform: template_id}}``, or None when unreadable
TemplatesFor = Callable[[str], Optional[Mapping[int, Mapping[str, str]]]]


def _utc(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts)) if ts else None
    except ValueError:
        return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) if dt else None


def snapshot_day7(short, platform: str, counter: str) -> Optional[Tuple[float, datetime]]:
    """``(value, fetched_at)`` of the measured ``platform`` observation nearest
    age 7d reporting ``counter``, or None when none lies within 12h of it."""
    best = None
    for entry in short.observation_log.get(platform) or []:
        if entry.get("status") != ObservationStatus.MEASURED:
            continue
        value = (entry.get("cumulative") or {}).get(counter)
        event, fetched = _utc(entry.get("event_at")), _utc(entry.get("fetched_at"))
        if value is None or event is None or fetched is None:
            continue
        miss = abs(fetched - event - SNAPSHOT_TARGET)
        if miss <= SNAPSHOT_TOLERANCE and (best is None or miss < best[0]):
            best = (miss, (value, fetched))
    return best[1] if best else None


def caption_evidence(
    records: Iterable[Any], templates_for: TemplatesFor, now: datetime
) -> CaptionEvidence:
    """Waypost ``CaptionEvidence`` from tracking records; ``now`` is ``as_of``."""
    values: Dict[Tuple[str, str], List[Tuple[float, datetime]]] = {}
    for record in records:
        tagged = templates_for(record.project) or {}
        for short in record.shorts:
            openers = tagged.get(short.index) or {}
            for platform, (counter, _source) in METRICS.items():
                template_id = openers.get(platform)
                if not template_id or not short.platform_ids.get(platform):
                    continue
                snap = snapshot_day7(short, platform, counter)
                if snap is not None:
                    values.setdefault((platform, template_id), []).append(snap)
    observations = tuple(
        CaptionObservation(
            platform=platform,
            template_id=template_id,
            metric_source=METRICS[platform][1],
            median=float(statistics.median(v for v, _ in snaps)),
            n=len(snaps),
            window_start=min(t for _, t in snaps).isoformat(),
            window_end=max(t for _, t in snaps).isoformat(),
        )
        for (platform, template_id), snaps in sorted(values.items())
    )
    return CaptionEvidence(
        as_of=now.astimezone(timezone.utc).isoformat(), observations=observations
    )


def write_caption_evidence(storage: Path, evidence: CaptionEvidence) -> Path:
    """Write ``evidence`` to ``<storage>/caption_evidence.json`` atomically."""
    path = Path(storage) / EVIDENCE_FILE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{EVIDENCE_FILE}.")
    with os.fdopen(fd, "w") as fh:
        json.dump(asdict(evidence), fh, indent=2)
    os.replace(tmp, path)
    return path


def load_caption_evidence(storage: Path) -> Optional[CaptionEvidence]:
    """The evidence file, or None when absent or unreadable (logged)."""
    path = Path(storage) / EVIDENCE_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return CaptionEvidence(
            as_of=raw["as_of"],
            observations=tuple(CaptionObservation(**o) for o in raw.get("observations") or ()),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("caption evidence: cannot read %s: %s", path, exc)
        return None


def evidence_report_lines(evidence: CaptionEvidence) -> List[str]:
    """One ``CAPTION_EVIDENCE ...`` line per Meta platform for the nightly output."""
    lines = []
    for platform, (_counter, source) in METRICS.items():
        mine = [o for o in evidence.observations if o.platform == platform]
        lines.append(
            f"CAPTION_EVIDENCE platform={platform} metric={source} openers={len(mine)} "
            f"posts={sum(o.n for o in mine)}"
        )
    return lines
