"""Console SEO tab: the performance half.

The tab has been called "SEO & Performance" since it was built and showed only
the corrections ledger -- no performance at all. The data was there the whole
time: `youtube_sync` writes a PERFORMANCE_UPDATED event per video into
`seo_manager`'s event store, and `AnalyticsTracker` already aggregates it.

This module is the read, and it owns exactly one decision: what reaches the
template. Every number and every label comes from the tracker unchanged --
above all the `None`s. A console reader is the last place a missing rate can
quietly become a `0`, which reads as the real measurement "nobody clicked", so
nothing here defaults, rounds or coalesces (the tracker itself no longer does that
either).

It degrades rather than raises, twice over, the shape `schedule_view` uses:

* `seo_manager` is an OPTIONAL dependency of this service. Every other caller
  imports it lazily inside the function for that reason, and so does this.
* A fresh install has no event log. That is an empty report plus a sentence
  saying so -- never an error page, and never a silent empty table, which reads
  as "every video has zero views".

Constructing an `SEOEventStore` mkdirs its storage path, so the log's existence
is checked BEFORE one is built: looking at a tab must not write to disk.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from seo_manager.analytics.anomaly_stats import Refusal

from ..config import Config
from ..tracking.anomaly import AnomalyFinding, check_feasibility, detect_post_anomaly
from ..tracking.channel_daily import load_rows as load_channel_daily_rows
from ..tracking.decline import platforms_seen
from ..tracking.store import TrackingStore  # module scope, not lazy, so tests can patch it
from ..tracking.suppression import classify_youtube_shorts_feed
from .settings import current_brand

logger = logging.getLogger(__name__)

#: Default for the site the pipeline stamps on every event it writes
#: (`longform_metadata`, `content_gates`), used only when a caller passes no
#: `config` at all. `performance_view` otherwise reads `config.seo.site_id` --
#: Waypost is multi-tenant, so hardcoding this would show an operator some
#: other brand's numbers.
SITE_ID = "gomoveshift"

#: Rows on the tab. The tracker's own top-performers list is capped at 10 and
#: this is the cap that matters -- a longer table is not a longer answer to
#: "which videos are working".
TOP_N = 10

#: What each `ctr_source` tag means in words, because the tag alone is the one
#: thing an operator must not have to guess at. A card click rate counts
#: in-player info-card clicks by viewers ALREADY watching; it says nothing
#: about whether a thumbnail won the click, and the rate that would (the
#: thumbnail impression CTR in YouTube Studio) is not exposed by the API at
#: all. A tag with no entry here falls back to the tag itself rather than
#: being dropped -- an unrecognised label is still evidence of something.
SOURCE_LABELS = {
    "youtube_card_click_rate": "card click rate (in-player, not thumbnail)",
    "youtube_thumbnail_impression_ctr": "thumbnail click-through rate",
    "unknown": "source not declared",
}

#: What each `ctr_basis` means. `ctr_source` says what the rate was measured
#: AGAINST; this says how the number in hand was arrived at.
BASIS_LABELS = {
    "clicks_over_impressions": "computed from counted clicks and impressions",
    "reported_rate": "as reported by the platform",
}

_EMPTY: Dict[str, Any] = {
    "rows": [],
    "hint": "",
    # Population the ten rows above are a slice of, and the cap that produced
    # the slice. `None` means "we never got as far as reading it" -- distinct
    # from 0, which would claim a definite count of no performance data.
    "total_content_with_performance": None,
    "top_performers_cap": TOP_N,
    # Site-wide totals across every recorded video, not just the ten shown.
    # `None` on every early-return path, same reasoning as above.
    "totals": None,
    # Page-level freshness: the newest `last_updated` and the oldest
    # `first_updated` among the rows actually shown. `None` on every
    # early-return path -- there are no rows to have read a timestamp off of.
    "data_as_of": None,
    "oldest_sample": None,
}


def _unknowns_for_early_return(hint: str) -> list:
    """On every path above, `hint` already says the one thing this read
    does not know -- there is no row/basis/population to inspect yet, so the
    trust panel has nothing to add beyond restating it."""
    return [hint] if hint else []


def _unknowns_for_rows(rows: list, totals: Optional[Dict[str, Any]], total_content) -> list:
    """The "what we do not know" panel, built from the exact state this
    function already computed -- never static prose that could drift from the
    numbers beside it. Each line names an actual `None`/`"unknown"` this read
    hit, not a guess about what might be missing."""
    notes = []
    if total_content is None:
        notes.append(
            "How many videos in total have recorded performance is unknown -- "
            "this install's tracker did not report a population count."
        )
    if totals is not None and totals.get("ctr") is None:
        notes.append("The channel-wide click rate has not been measured.")
    for row in rows:
        if row["ctr"] is None:
            notes.append(f"{row['content_id']}: click rate not measured.")
        elif row["ctr_source"] == "unknown":
            notes.append(f"{row['content_id']}: click rate's source is not declared.")
    return notes


def _unknowns_badge(unknowns: list) -> str:
    """One line for the badge near the tab header -- the panel itself is the
    detail, this is the summary that tells an operator whether to open it."""
    if not unknowns:
        return "Nothing unknown about these numbers right now."
    plural = "s" if len(unknowns) != 1 else ""
    return f"{len(unknowns)} thing{plural} we do not know about these numbers."


def events_path(config: Optional[Any] = None) -> Path:
    """The same `<storage_path>/<site_id>/events.jsonl` the pipeline writes,
    resolved through `SEOConfig.events_jsonl` rather than rebuilt here."""
    config = config if config is not None else Config.load()
    return Path(config.seo.events_jsonl)


def performance_view(config: Optional[Any] = None) -> Dict[str, Any]:
    """Top videos by views, each with its rate, what that rate measures, its
    denominator, and when we last heard anything about it.

    A `hint` is set whenever `rows` is empty for a reason the operator needs --
    no log yet, the optional dependency missing, a log we could not read. It is
    a sentence on the page, not an exception: the corrections editor beside it
    still works, and a performance failure must not take the tab down with it.

    also returns page-level `data_as_of` (the newest `last_updated`
    among the rows shown) and `oldest_sample` (the oldest `first_updated`
    among them), each `None` when no row carries that timestamp. No staleness
    threshold or verdict is computed here -- only what was actually observed.
    """
    config = config if config is not None else Config.load()
    log = events_path(config)
    if not log.exists():
        # Checked before any store is constructed: `SEOEventStore.__init__`
        # mkdirs, and a tab that creates the directory it reads would make
        # "nothing recorded yet" indistinguishable from "recorded nothing".
        hint = f"No performance data yet: nothing has been recorded at {log}."
        return {
            **_EMPTY,
            "hint": hint,
            "unknowns": _unknowns_for_early_return(hint),
            "unknowns_badge": _unknowns_badge(_unknowns_for_early_return(hint)),
        }

    try:
        from seo_manager.analytics.tracker import AnalyticsTracker
        from seo_manager.storage.event_store import SEOEventStore
    except ImportError as exc:
        logger.warning("console seo tab: performance unavailable: %s", exc)
        hint = (
            "Performance is unavailable because seo_manager is not installed "
            f"in this environment ({exc})."
        )
        return {
            **_EMPTY,
            "hint": hint,
            "unknowns": _unknowns_for_early_return(hint),
            "unknowns_badge": _unknowns_badge(_unknowns_for_early_return(hint)),
        }

    try:
        site_id = getattr(config.seo, "site_id", SITE_ID)
        tracker = AnalyticsTracker(store=SEOEventStore(log.parent), site_id=site_id)
        summary = tracker.get_summary()
    except Exception as exc:  # noqa: BLE001 -- any read failure degrades to empty+hint
        logger.warning("console seo tab: performance read failed: %s", exc)
        hint = f"Performance data could not be read: {exc}"
        return {
            **_EMPTY,
            "hint": hint,
            "unknowns": _unknowns_for_early_return(hint),
            "unknowns_badge": _unknowns_badge(_unknowns_for_early_return(hint)),
        }

    rows = [_row(row) for row in summary.get("top_performers", [])[:TOP_N]]
    if not rows:
        # The log exists and yielded nothing. That is NOT the same as no log,
        # and it must not render as a bare empty table: the store skips lines
        # it cannot parse with only a warning, so "every video has zero views"
        # and "we could not read any of this file" look identical on the page
        # unless the page says which.
        hint = (
            f"No performance data could be read from {log}. The file exists, so "
            "either nothing has been recorded for this site yet or its lines "
            "could not be parsed -- check the log for skipped-line warnings."
        )
        return {
            **_EMPTY,
            "hint": hint,
            "unknowns": _unknowns_for_early_return(hint),
            "unknowns_badge": _unknowns_badge(_unknowns_for_early_return(hint)),
        }
    # `.get(...)` throughout: an older seo_manager without the unknowns keys must
    # degrade to "unknown", never crash and never invent a number by
    # falling back to `len(rows)` -- that would state the cap as if it
    # were the population, the exact bug this exists to kill.
    total_content_with_performance = summary.get("performance_metrics", {}).get(
        "total_content_with_performance"
    )
    totals = _totals(summary.get("performance_metrics", {}))
    unknowns = _unknowns_for_rows(rows, totals, total_content_with_performance)
    # Page-level freshness, over the rows actually shown -- never the
    # full population, which this read never sees. `max`/`min` over ISO-8601
    # strings sorts correctly the same way `_aggregate` already relies on
    # (tracker.py); a row with no timestamp contributes to neither.
    last_updates = [row["last_updated"] for row in rows if row["last_updated"]]
    first_updates = [row["first_updated"] for row in rows if row["first_updated"]]
    return {
        "rows": rows,
        "hint": "",
        "total_content_with_performance": total_content_with_performance,
        "top_performers_cap": summary.get("performance_metrics", {}).get(
            "top_performers_cap", TOP_N
        ),
        "totals": totals,
        "unknowns": unknowns,
        "unknowns_badge": _unknowns_badge(unknowns),
        "data_as_of": max(last_updates) if last_updates else None,
        "oldest_sample": min(first_updates) if first_updates else None,
    }


def _totals(performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """The site-wide numbers `get_summary` pools over every recorded event,
    on the same never-invent-a-0 contract as `_row`: `ctr` is `None` when
    unmeasured, and `ctr_basis_label` is the empty string in that case rather
    than a bare basis tag the template would have to translate itself."""
    ctr = performance_metrics.get("avg_ctr")
    basis = performance_metrics.get("avg_ctr_basis")
    return {
        "total_views": performance_metrics.get("total_views"),
        "total_clicks": performance_metrics.get("total_clicks"),
        "total_impressions": performance_metrics.get("total_impressions"),
        "ctr": ctr,
        "ctr_percent": None if ctr is None else round(ctr * 100, 2),
        "ctr_basis": basis,
        "ctr_basis_label": BASIS_LABELS.get(basis, ""),
    }


def _observation_window_days(
    first_updated: Optional[str], last_updated: Optional[str]
) -> Optional[int]:
    """Whole days between the tracker's own first- and last-seen timestamps
    for one video. `None` when either timestamp is missing or unparseable --
    never a fabricated `0`, which would read as "we watched this video for
    zero days" rather than "we don't know how long we've been watching it"."""
    if not first_updated or not last_updated:
        return None
    try:
        first = datetime.fromisoformat(first_updated)
        last = datetime.fromisoformat(last_updated)
    except (TypeError, ValueError):
        return None
    return (last - first).days


def _row(performer: Dict[str, Any]) -> Dict[str, Any]:
    """One tracker row, plus the two labels the template must not invent.

    `ctr` is passed through exactly -- `None` included. `ctr_percent` is a
    convenience for the template and is `None` on the same rows, so a missing
    rate has no numeric form anywhere for a `{{ x or 0 }}` to find.

    `first_updated` is the tracker's own earliest-sample timestamp for
    this video, on the same never-invent contract as `last_updated`.
    `observation_window_days` is derived from the two -- whole days, `None`
    when either is missing -- rather than passed through, because the tracker
    does not compute it."""
    ctr = performer.get("ctr")
    source = performer.get("ctr_source") or "unknown"
    basis = performer.get("ctr_basis")
    first_updated = performer.get("first_updated")
    last_updated = performer.get("last_updated")
    return {
        "content_id": performer.get("content_id"),
        "views": performer.get("views"),
        "clicks": performer.get("clicks"),
        "impressions": performer.get("impressions"),
        "ctr": ctr,
        "ctr_percent": None if ctr is None else round(ctr * 100, 2),
        "ctr_source": source,
        "ctr_source_label": SOURCE_LABELS.get(source, source),
        "ctr_basis": basis,
        "ctr_basis_label": BASIS_LABELS.get(basis, ""),
        "first_updated": first_updated,
        "last_updated": last_updated,
        "observation_window_days": _observation_window_days(first_updated, last_updated),
        "data_points": performer.get("data_points"),
    }


# The already-built, shadow-only hypothesis engines
# (`tracking.anomaly`/`tracking.suppression`) had zero console call sites --
# a cron-only stdout report nobody but an operator tailing a log ever saw.
# This is the read, on the same never-raise/never-invent-a-number contract
# as `performance_view` above. Neither engine is modified here or has its
# thresholds re-implemented: alpha comes from `anomaly.check_feasibility()`
# live, and the suppression thresholds live entirely inside `suppression.py`.

_EMPTY_ANOMALIES: Dict[str, Any] = {
    "findings": [],
    "refusals": [],
    "suppression": [],
    "hint": "",
    # False on every early-return path below, and only there -- see
    # `anomaly_view`'s docstring for what "no data" means here. `seo.html`
    # renders no anomalies section at all when this is False, rather than a
    # blank one: an install with nothing tracked yet is not the same claim
    # as "we checked and found nothing".
    "has_data": False,
}


def _finding_row(finding: AnomalyFinding) -> Dict[str, Any]:
    """One `AnomalyFinding`, plus `shadow`, which the dataclass itself does
    not carry as a field -- it is only ever stamped into `log_line()`'s
    string, and a template must not have to parse a log line to find it."""
    return {
        "platform": finding.platform,
        "video_id": finding.video_id,
        "kind": finding.kind,
        "target": finding.target,
        "ref_median": finding.ref_median,
        "deviation": finding.deviation,
        "relative_effect": finding.relative_effect,
        "p_value": finding.p_value,
        "n": finding.n,
        "allocated_alpha": finding.allocated_alpha,
        "shadow": True,
    }


def _refusal_row(refusal) -> Dict[str, Any]:
    """One `AnomalyRefusal`, same `shadow` treatment as `_finding_row` --
    the kernel refused before any statistic existed to carry it, so there
    is no `n`/`p_value` to show beyond what the refusal itself states."""
    return {
        "platform": refusal.platform,
        "reason": refusal.reason,
        "n": refusal.n,
        "required": refusal.required,
        "shadow": True,
    }


def _suppression_rows(brand: str) -> List[Dict[str, Any]]:
    """YouTube Shorts-feed suppression, translated from
    `classify_youtube_shorts_feed`'s `{day: Status}` map into the one thing
    worth showing: the latest day's status and how many consecutive days it
    has held. The classifier computes no statistic at all -- no n, no
    p-value, no deviation -- and this must never invent one for a status
    that carries none; that is the whole difference from `_finding_row`
    above.

    Facebook is not translated here: `channel_daily` now stores the Page
    daily series (``facebook:<metric>`` rows), but `classify_facebook_reel_reach`
    is not yet switched onto it (see `docs/facebook-metrics.md`).
    """
    rows = load_channel_daily_rows(brand)
    if not rows:
        return []
    day_states = classify_youtube_shorts_feed(rows)
    if not day_states:
        return []
    days = sorted(day_states)
    latest_day = days[-1]
    latest_status = day_states[latest_day]
    consecutive = 0
    for day in reversed(days):
        if day_states[day] != latest_status:
            break
        consecutive += 1
    return [
        {
            "platform": "youtube",
            "as_of": latest_day,
            "status": latest_status,
            "consecutive_days": consecutive,
            "shadow": True,
        }
    ]


def anomaly_view(config: Optional[Any] = None) -> Dict[str, Any]:
    """Hypothesis-grade per-post dip/spike findings and per-platform
    suppression status, read-only and shadow only -- nothing in this
    service acts on any of it, same as the CLI it is a second surface for.

    Never raises: any read failure (a corrupt tracking record, a broken
    channel-daily store) degrades to the same empty, section-absent shape
    as no data at all, logged rather than shown -- an operator's tab must
    not go down over a report nothing else in the pipeline depends on.

    `has_data` is `False` only when there is truly nothing to read: no
    platform has a tracked short (`tracking.decline.platforms_seen`) AND no
    channel-daily rows exist for the current brand. Once either source has
    something, the section renders even with zero current findings -- a
    page that goes quiet the moment nothing is anomalous would look
    identical to a page that never checked, which is the same lie
    `performance_view` refuses to tell about an unmeasured rate.

    `config` is accepted for the same call shape as `performance_view` but
    is currently unused: neither data source here is keyed by `SEOConfig`
    (`TrackingStore` reads `GMS_OUTPUT_DIR`, `channel_daily` reads the
    console's own `current_brand()`), so there is nothing in it for this
    read to consult yet.
    """
    try:
        records = TrackingStore().load_all_records()
        brand = current_brand()
        suppression = _suppression_rows(brand)
        platforms = platforms_seen(records)
        if not platforms and not suppression:
            return dict(_EMPTY_ANOMALIES)

        findings: List[Dict[str, Any]] = []
        refusals: List[Dict[str, Any]] = []
        hint = ""
        if platforms:
            feasible = check_feasibility()
            if isinstance(feasible, Refusal):
                hint = (
                    "Post-anomaly detection is not feasible under the current "
                    f"policy: {feasible.reason}"
                )
            else:
                for platform in platforms:
                    result = detect_post_anomaly(records, platform, feasible.allocated_alpha)
                    if result is None:
                        continue
                    if isinstance(result, AnomalyFinding):
                        findings.append(_finding_row(result))
                    else:
                        refusals.append(_refusal_row(result))

        return {
            "findings": findings,
            "refusals": refusals,
            "suppression": suppression,
            "hint": hint,
            "has_data": True,
        }
    except Exception as exc:  # noqa: BLE001 -- degrade to no-data, never take the tab down
        logger.warning("console seo tab: anomalies unavailable: %s", exc)
        return dict(_EMPTY_ANOMALIES)
