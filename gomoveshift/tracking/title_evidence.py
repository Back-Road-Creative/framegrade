"""YouTube title evidence: how each shipped title template's videos did on YouTube.

Waypost's ``analytics.title_evidence`` turns collected ``youtube_reach_combined``
rows (impressions and thumbnail click-through per video) into per-template rates,
given a map of YouTube video id to the template and format that titled it. This
module builds that map from the stores that hold the shipped decisions:

* a **Short**: ``shorts/lf_*/social_posts.json`` carries each post's
  ``title_decision`` (the publish plan sidecar does not); the tracking store's
  ``ShortRecord.platform_ids["youtube"]`` carries the video id;
* a **long-form**: ``youtube_uploads.json`` entries carry ``title_decision`` beside
  the ``video_id``.

A decision with no chosen template id, or whose reason is ``facts_unavailable``
(the baseline shipped with nothing behind it), is not evidence. Nothing here ranks:
``seo_manager.analytics.title_selection`` decides what the numbers mean. The
nightly ``metrics collect`` writes the result to ``<seo storage>/title_evidence.json``;
every shipped-title caller reads it back with :func:`load_title_evidence`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from seo_manager.analytics.title_evidence import (
    TitleAttribution,
    TitleEvidenceRefused,
    TitleEvidenceResult,
    snapshot_from_dict,
    snapshot_to_dict,
)
from seo_manager.analytics.title_selection import DEFAULT_POLICY, EvidenceSnapshot

logger = logging.getLogger(__name__)

EVIDENCE_FILE = "title_evidence.json"

#: Waypost tenant the reach collector writes this channel's rows under
#: (``_ops/cron/seo-collect.cron`` passes ``--tenant gomoveshift``).
TENANT = "gomoveshift"

FACTS_UNAVAILABLE = "facts_unavailable"

#: project -> that project's pipeline output dir, or None when unknown
BaseFor = Callable[[str], Optional[Path]]


def chosen_template(decision: Any) -> Optional[str]:
    """The template id a shipped ``title_decision`` chose, or None when it names
    none or the decision rested on no facts."""
    if not isinstance(decision, dict) or decision.get("reason") == FACTS_UNAVAILABLE:
        return None
    for candidate in decision.get("candidates") or ():
        if isinstance(candidate, dict) and candidate.get("index") == decision.get("chosen_index"):
            template_id = candidate.get("template_id")
            return template_id if isinstance(template_id, str) and template_id else None
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("title evidence: cannot read %s: %s", path, exc)
        return None


def _short_decisions(base: Path) -> Dict[int, Any]:
    """``{short_index: title_decision}`` from every LF's ``social_posts.json``."""
    out: Dict[int, Any] = {}
    for path in sorted(base.glob("shorts/lf_*/social_posts.json")):
        posts = _read_json(path)
        for post in posts if isinstance(posts, list) else ():
            if isinstance(post, dict) and isinstance(post.get("short_index"), int):
                out.setdefault(post["short_index"], post.get("title_decision"))
    return out


def title_attribution(records: Iterable[Any], base_for: BaseFor) -> Dict[str, TitleAttribution]:
    """``{youtube video id: TitleAttribution}`` for every tracked project whose
    output dir ``base_for`` resolves. An unreadable file costs that file only."""
    out: Dict[str, TitleAttribution] = {}
    for record in records:
        base = base_for(record.project)
        if base is None:
            continue
        decisions = _short_decisions(base)
        for short in record.shorts:
            video_id = short.platform_ids.get("youtube")
            template_id = chosen_template(decisions.get(short.index))
            if video_id and template_id:
                out[video_id] = TitleAttribution(template_id, "short")
        uploads = _read_json(base / "youtube_uploads.json")
        for entry in uploads.values() if isinstance(uploads, dict) else ():
            if not isinstance(entry, dict) or not entry.get("video_id"):
                continue
            template_id = chosen_template(entry.get("title_decision"))
            if template_id:
                out[str(entry["video_id"])] = TitleAttribution(template_id, "longform")
    return out


def write_title_evidence(storage: Path, result: TitleEvidenceResult) -> Path:
    """Write ``result.snapshot`` to ``<storage>/title_evidence.json`` atomically."""
    path = Path(storage) / EVIDENCE_FILE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{EVIDENCE_FILE}.")
    with os.fdopen(fd, "w") as fh:
        json.dump(snapshot_to_dict(result.snapshot), fh, indent=2)
    os.replace(tmp, path)
    return path


def load_title_evidence(storage: Path) -> Optional[EvidenceSnapshot]:
    """The evidence snapshot, or None when absent, unreadable or refused (logged)."""
    path = Path(storage) / EVIDENCE_FILE
    if not path.exists():
        return None
    try:
        return snapshot_from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, TitleEvidenceRefused) as exc:
        logger.warning("title evidence: cannot use %s: %s", path, exc)
        return None


def title_evidence_report_lines(result: TitleEvidenceResult) -> List[str]:
    """One ``TITLE_EVIDENCE ...`` line per format and its policy traffic source,
    then one line of row counters, for the nightly output."""
    lines = []
    for fmt, traffic in DEFAULT_POLICY.traffic_by_format.items():
        mine = [
            o
            for o in result.snapshot.observations
            if o.format == fmt and o.traffic_source == traffic
        ]
        lines.append(
            f"TITLE_EVIDENCE format={fmt} traffic={traffic} templates={len(mine)} "
            f"impressions={sum(o.n for o in mine)}"
        )
    lines.append(
        f"TITLE_EVIDENCE_ROWS used={result.rows_used} unattributed={result.rows_unattributed} "
        f"unparseable_subject={result.rows_unparseable_subject} "
        f"other_traffic={result.rows_other_traffic} unknown_value={result.rows_unknown_value} "
        f"out_of_window={result.rows_out_of_window} "
        f"dropped_rate_over_1={result.groups_dropped_rate_over_1}"
    )
    return lines
