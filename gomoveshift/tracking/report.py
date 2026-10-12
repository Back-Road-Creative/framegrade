"""
Performance Reporter — generates human-readable markdown reports.

Analyses tracking data to find:
- Best/worst performing shorts
- Posting time vs performance correlations
- Weather vs performance correlations
- Hook text vs performance correlations
- Actionable recommendations for the next batch
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from .observation import ObservationStatus
from .schema import ShortRecord, TrackingRecord
from .store import TrackingStore

logger = logging.getLogger(__name__)


def _refusals_by_platform(record: TrackingRecord) -> Dict[str, Dict[str, Any]]:
    """Per-platform tally of the most recent non-measurement per short.

    A refusal is a fact about the world -- no credentials, a missing scope, a
    post the platform would not return -- and each short's ``observation_log``
    already holds it. Nothing read that log, so a run where every platform
    refused produced a report indistinguishable from a run that never
    happened. Only the LAST observation per (short, platform) counts: a short
    refused this morning and measured this afternoon is measured.
    """
    tally: Dict[str, Dict[str, Any]] = {}
    for short in record.shorts:
        for platform, series in (short.observation_log or {}).items():
            if not series:
                continue
            latest = series[-1]
            status = latest.get("status")
            if status not in ObservationStatus.UNANSWERED:
                continue
            bucket = tally.setdefault(platform, {})
            entry = bucket.setdefault(status, {"count": 0, "detail": ""})
            entry["count"] += 1
            if not entry["detail"] and latest.get("detail"):
                entry["detail"] = latest["detail"]
    return tally


def _refusals_section(record: TrackingRecord) -> str:
    """Render the refusals, or the empty string when there were none.

    Deliberately reports no numbers: a refusal carries none, and rendering it
    beside measured figures is exactly the conflation this section exists to
    end.
    """
    tally = _refusals_by_platform(record)
    if not tally:
        return ""
    attempted = len(record.shorts_collection_attempted)
    lines = [
        "## Collection refusals",
        "",
        f"A collection run reached {attempted} of {record.total_shorts} short(s); "
        "the platforms below refused rather than reporting numbers. These are "
        "recorded, not measured zeros -- the full series is in each short's "
        "observation log.",
        "",
    ]
    for platform in sorted(tally):
        for status in sorted(tally[platform]):
            entry = tally[platform][status]
            detail = f" — {entry['detail']}" if entry["detail"] else ""
            lines.append(f"- **{platform}** — {status}: {entry['count']} short(s){detail}")
    return "\n".join(lines) + "\n"


def _known_engagements(shorts: List[ShortRecord]) -> List[float]:
    """Engagement rates that are actually known, dropping unknown (None) ones."""
    return [e for s in shorts if (e := s.total_engagement) is not None]


#: ``TrackingRecord`` (schema.py) enumerates only ``shorts``
#: — there is no record anywhere in ``tracking/`` for a long-form video, a
#: still, a carousel, or a post a human published directly in a platform's
#: app rather than through this pipeline. Every figure below (totals,
#: averages, best/worst rankings) is computed over that shorts-only
#: population. Without this line a reader has no way to tell a shorts-only
#: total from a channel-wide one — printed here on every report surface
#: rather than left to be inferred from a column header.
COVERAGE_NOTE = (
    "**Coverage:** shorts registered with this tracker only. Long-form "
    "videos, stills, carousels, and posts published directly in a "
    "platform's app (not through this pipeline) are not enumerated here — "
    "absent from these figures, not counted as zero."
)


@dataclass
class EngagementBasis:
    """A mean engagement rate together with what it was computed over.

    ``ShortRecord.total_engagement`` is already a per-short blend across
    that short's OWN platforms (``schema.py``'s business, unchanged here).
    The question this type answers is different: is it safe to average that
    per-short rate ACROSS shorts? Two shorts posted to different platform
    combinations carry rates built from different platforms' definitions of
    a view/like/share — the same reason ``observation.compare_views``
    refuses to blend raw view counts across platforms. This consumer only
    ever sees a flat ``List[ShortRecord]``, with no independent per-platform
    post-format or per-observation fetch timestamp to bucket on the way
    ``compare_views`` also does for format and age — so it declines strictly
    on platform-combination mismatch, the one dimension of that invariant it
    can actually see here, rather than guessing a format or age bucket.

    ``value`` is only ever a real number when every short folded in shares
    the same ``platforms``; otherwise it is ``None`` and ``declined_reason``
    says why. A separate empty-vs-declined distinction: "no shorts had a
    known rate" and "shorts had rates but weren't comparable" are different
    facts and must not render as the same blank state.
    """

    value: Optional[float]
    sample_size: int
    platforms: FrozenSet[str] = field(default_factory=frozenset)
    declined_reason: str = ""


def _avg_engagement(shorts: List[ShortRecord]) -> EngagementBasis:
    """Mean engagement over shorts with a known rate, refusing to blend
    across platform combinations. See ``EngagementBasis``."""
    known = [s for s in shorts if s.total_engagement is not None]
    if not known:
        return EngagementBasis(
            value=None, sample_size=0, declined_reason="no shorts with a known engagement rate"
        )
    platform_sets = {frozenset(s.metrics.keys()) for s in known}
    if len(platform_sets) > 1:
        described = ", ".join(
            "+".join(sorted(p)) if p else "no-platform" for p in sorted(platform_sets, key=sorted)
        )
        return EngagementBasis(
            value=None,
            sample_size=len(known),
            declined_reason=(
                "engagement rates from shorts posted to different platform "
                f"combinations ({described}) are not averaged: each combination "
                "blends a different set of platform-defined rates, the same "
                "reason compare_views refuses to blend view counts across platforms"
            ),
        )
    (platforms,) = platform_sets
    rate = statistics.mean(s.total_engagement for s in known)
    return EngagementBasis(value=rate, sample_size=len(known), platforms=platforms)


def _fmt_engagement(value: Optional[float]) -> str:
    """Render an engagement rate, or the file's existing "—" for unknown."""
    return f"{value:.2%}" if value is not None else "—"


def _fmt_engagement_basis(basis: EngagementBasis) -> str:
    """Render an ``EngagementBasis`` — "—" whether nothing was known or the
    shorts weren't comparable. Callers wanting the reason read
    ``basis.declined_reason`` directly."""
    return _fmt_engagement(basis.value)


class PerformanceReporter:
    """
    Generates performance insight reports from tracking data.

    Reports are markdown-formatted for human readability.
    """

    def __init__(self, store: Optional[TrackingStore] = None):
        self.store = store or TrackingStore()

    # ── Main Report Generation ───────────────────────────────────────────

    def generate_project_report(self, project: str) -> str:
        """
        Generate a full performance report for a single project.

        Args:
            project: Project name

        Returns:
            Markdown-formatted report string
        """
        record = self.store.load(project)
        shorts = record.shorts_with_metrics

        refusals = _refusals_section(record)

        if not shorts:
            attempted = len(record.shorts_collection_attempted)
            if attempted:
                # Reached and refused is not the same fact as never asked.
                opening = (
                    f"⚠️ Nothing measured. A collection run reached {attempted} of "
                    f"{len(record.shorts)} registered shorts and no platform reported "
                    f"any numbers.\n"
                )
            else:
                opening = (
                    f"⚠️ No metrics collected yet for {len(record.shorts)} registered shorts.\n"
                    f"Run `collector.collect_all(record)` after shorts are published.\n"
                )
            return (
                f"# Performance Report: {project}\n\n"
                f"**Location:** {record.location}\n\n"
                f"{opening}"
                f"{COVERAGE_NOTE}\n" + (f"\n{refusals}" if refusals else "")
            )

        sections = [
            self._header(record),
            self._summary(shorts),
            self._best_worst(shorts),
            self._posting_time_analysis(shorts),
            self._weather_analysis(shorts),
            self._hook_analysis(shorts),
            self._duration_analysis(shorts),
            self._platform_breakdown(shorts),
            self._recommendations(shorts, record),
        ]
        if refusals:
            sections.append(refusals)

        return "\n\n".join(sections)

    def generate_cross_project_report(self) -> str:
        """
        Generate a report spanning all tracked projects.

        Useful for finding patterns across different locations/seasons.
        """
        all_shorts = self.store.get_all_shorts()
        # Measured, not merely attempted: ``collected_at`` is
        # stamped on a run every platform refused, and a refusal carries no
        # numbers to report. Mirrors TrackingRecord.shorts_with_metrics.
        tracked = [s for s in all_shorts if s.metrics]

        if not tracked:
            return (
                "# Cross-Project Report\n\n"
                "⚠️ No metrics collected across any project yet.\n"
                f"{COVERAGE_NOTE}\n"
            )

        projects = self.store.list_projects()

        sections = [
            f"# Cross-Project Performance Report\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"**Projects:** {len(projects)}\n"
            f"**Total shorts with metrics:** {len(tracked)}\n"
            f"{COVERAGE_NOTE}\n",
            self._summary(tracked),
            self._best_worst(tracked, label="All Projects"),
            self._posting_time_analysis(tracked),
            self._weather_analysis(tracked),
            self._hook_analysis(tracked),
            self._duration_analysis(tracked),
            self._recommendations(tracked),
        ]

        return "\n\n".join(sections)

    def save_report(self, project: str, output_path: Optional[Path] = None) -> Path:
        """Generate and save a report to disk."""
        report = self.generate_project_report(project)

        if output_path is None:
            output_path = self.store.root / project / "tracking" / f"{project}_report.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Saved report to %s", output_path)
        return output_path

    # ── Report Sections ──────────────────────────────────────────────────

    def _header(self, record: TrackingRecord) -> str:
        tracked = record.shorts_with_metrics
        return (
            f"# Performance Report: {record.project}\n\n"
            f"**Location:** {record.location}\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"**Total shorts:** {record.total_shorts}\n"
            f"**Shorts with metrics:** {len(tracked)}\n"
            f"**Created:** {record.created_at[:10] if record.created_at else 'N/A'}\n"
            f"{COVERAGE_NOTE}\n"
        )

    def _summary(self, shorts: List[ShortRecord]) -> str:
        views = [s.total_views for s in shorts]

        if not views:
            return "## Summary\n\nNo data available.\n"

        engagement_basis = _avg_engagement(shorts)
        table = (
            f"## Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total views | {sum(views):,} |\n"
            f"| Average views/short | {statistics.mean(views):,.0f} |\n"
            f"| Median views/short | {statistics.median(views):,.0f} |\n"
            f"| Best views | {max(views):,} |\n"
            f"| Worst views | {min(views):,} |\n"
            f"| Avg engagement rate | {_fmt_engagement_basis(engagement_basis)} |\n"
            f"| Shorts analysed | {len(shorts)} |\n"
        )
        if engagement_basis.declined_reason:
            table += f"\n_Avg engagement rate: {engagement_basis.declined_reason}._\n"
        return table

    def _best_worst(self, shorts: List[ShortRecord], label: str = "") -> str:
        if not shorts:
            return ""

        by_views = sorted(shorts, key=lambda s: s.total_views, reverse=True)

        title = f"## Best & Worst Performing{' — ' + label if label else ''}"

        lines = [title, ""]

        # Top 5
        lines.append("### 🏆 Top Performers")
        lines.append("")
        lines.append("| Rank | Short | Views | Engagement | Hook | Weather | Posted |")
        lines.append("|------|-------|-------|------------|------|---------|--------|")
        for i, s in enumerate(by_views[:5], 1):
            posted = s.posted_at[:10] if s.posted_at else "—"
            hour = f"{s.posting_hour}:00" if s.posting_hour is not None else "—"
            lines.append(
                f"| {i} | #{s.index} ({s.file or '—'}) | {s.total_views:,} | "
                f"{_fmt_engagement(s.total_engagement)} | {s.content.hook or '—'} | "
                f"{s.content.weather or '—'} | {posted} {hour} |"
            )

        lines.append("")

        # Bottom 3
        if len(by_views) > 3:
            lines.append("### 📉 Lowest Performers")
            lines.append("")
            lines.append("| Short | Views | Engagement | Hook | Weather | Posted |")
            lines.append("|-------|-------|------------|------|---------|--------|")
            for s in by_views[-3:]:
                posted = s.posted_at[:10] if s.posted_at else "—"
                hour = f"{s.posting_hour}:00" if s.posting_hour is not None else "—"
                lines.append(
                    f"| #{s.index} ({s.file or '—'}) | {s.total_views:,} | "
                    f"{_fmt_engagement(s.total_engagement)} | {s.content.hook or '—'} | "
                    f"{s.content.weather or '—'} | {posted} {hour} |"
                )

        return "\n".join(lines)

    def _posting_time_analysis(self, shorts: List[ShortRecord]) -> str:
        """Analyse performance by posting hour."""
        by_hour: Dict[int, List[int]] = defaultdict(list)

        for s in shorts:
            hour = s.posting_hour
            if hour is not None:
                by_hour[hour].append(s.total_views)

        if not by_hour:
            return "## Posting Time Analysis\n\nNo posting time data available.\n"

        lines = [
            "## ⏰ Posting Time Analysis",
            "",
            "| Time Slot | Shorts | Avg Views | Total Views |",
            "|-----------|--------|-----------|-------------|",
        ]

        # Group into time slots
        slots = {
            "Early Morning (6-9)": list(range(6, 9)),
            "Morning (9-12)": list(range(9, 12)),
            "Afternoon (12-15)": list(range(12, 15)),
            "Late Afternoon (15-18)": list(range(15, 18)),
            "Evening (18-21)": list(range(18, 21)),
            "Night (21-24)": list(range(21, 24)),
        }

        best_slot = ""
        best_avg = 0

        for slot_name, hours in slots.items():
            slot_views = []
            for h in hours:
                slot_views.extend(by_hour.get(h, []))
            if slot_views:
                avg = statistics.mean(slot_views)
                if avg > best_avg:
                    best_avg = avg
                    best_slot = slot_name
                lines.append(
                    f"| {slot_name} | {len(slot_views)} | {avg:,.0f} | {sum(slot_views):,} |"
                )

        if best_slot:
            lines.append("")
            lines.append(f"**💡 Best time slot: {best_slot}** (avg {best_avg:,.0f} views)")

        return "\n".join(lines)

    def _weather_analysis(self, shorts: List[ShortRecord]) -> str:
        """Analyse performance by weather condition."""
        by_weather: Dict[str, List[ShortRecord]] = defaultdict(list)

        for s in shorts:
            weather = s.content.weather.lower().strip() if s.content.weather else "unknown"
            by_weather[weather].append(s)

        if not by_weather or (len(by_weather) == 1 and "unknown" in by_weather):
            return "## Weather Analysis\n\nNo weather data available.\n"

        lines = [
            "## 🌤️ Weather vs Performance",
            "",
            "| Weather | Shorts | Avg Views | Avg Engagement | Best Views |",
            "|---------|--------|-----------|----------------|------------|",
        ]

        best_weather = ""
        best_avg = 0

        for weather, group in sorted(by_weather.items()):
            views = [s.total_views for s in group]
            avg_v = statistics.mean(views)
            if avg_v > best_avg:
                best_avg = avg_v
                best_weather = weather
            lines.append(
                f"| {weather.title()} | {len(group)} | {avg_v:,.0f} | "
                f"{_fmt_engagement_basis(_avg_engagement(group))} | {max(views):,} |"
            )

        if best_weather and best_weather != "unknown":
            lines.append("")
            lines.append(f"**💡 Best weather: {best_weather.title()}** (avg {best_avg:,.0f} views)")

        return "\n".join(lines)

    def _hook_analysis(self, shorts: List[ShortRecord]) -> str:
        """Analyse performance by hook text / title format."""
        by_hook: Dict[str, List[ShortRecord]] = defaultdict(list)

        for s in shorts:
            hook = s.content.hook.strip() if s.content.hook else "no hook"
            by_hook[hook].append(s)

        if not by_hook or (len(by_hook) == 1 and "no hook" in by_hook):
            return "## Hook Analysis\n\nNo hook data available.\n"

        lines = [
            "## 🪝 Hook Text vs Performance",
            "",
            "| Hook | Shorts | Avg Views | Avg Engagement |",
            "|------|--------|-----------|----------------|",
        ]

        sorted_hooks = sorted(
            by_hook.items(),
            key=lambda kv: statistics.mean([s.total_views for s in kv[1]]),
            reverse=True,
        )

        for hook, group in sorted_hooks:
            views = [s.total_views for s in group]
            lines.append(
                f"| {hook} | {len(group)} | {statistics.mean(views):,.0f} | "
                f"{_fmt_engagement_basis(_avg_engagement(group))} |"
            )

        # Also check title formats if available
        by_format: Dict[str, List[ShortRecord]] = defaultdict(list)
        for s in shorts:
            fmt = s.content.title_format.strip() if s.content.title_format else ""
            if fmt:
                by_format[fmt].append(s)

        if by_format:
            lines.append("")
            lines.append("### Title Format Performance")
            lines.append("")
            lines.append("| Format | Shorts | Avg Views |")
            lines.append("|--------|--------|-----------|")
            for fmt, group in sorted(
                by_format.items(),
                key=lambda kv: statistics.mean([s.total_views for s in kv[1]]),
                reverse=True,
            ):
                views = [s.total_views for s in group]
                lines.append(f"| {fmt} | {len(group)} | {statistics.mean(views):,.0f} |")

        return "\n".join(lines)

    def _duration_bucket(self, duration_sec: float) -> str:
        """Fixed-width bucket label for a short's ``ContentMeta.duration_sec``.

        ``<= 0`` is "unknown" (never measured/never set), not folded into
        "≤15s" -- an unmeasured duration and a genuinely 3-second clip must
        not read the same in the table below.
        """
        if duration_sec <= 0:
            return "unknown"
        if duration_sec <= 15:
            return "≤15s"
        if duration_sec <= 30:
            return "16–30s"
        if duration_sec <= 60:
            return "31–60s"
        return ">60s"

    def _duration_analysis(self, shorts: List[ShortRecord]) -> str:
        """Analyse performance by ``ContentMeta.duration_sec`` bucket.

        Makes the length experiment measurable by cohorting shorts
        into fixed duration buckets and reporting per-bucket sample size and
        mean engagement, same "measured-only, never a silent zero" rule as
        every other section here (``_avg_engagement`` drops shorts with no
        known engagement rate rather than treating them as 0%).

        A ``duration_sec <= 0`` short (never measured/never set) buckets to
        "unknown" rather than being dropped -- silently excluding it would
        undercount the cohort denominator the same way a refusal must not
        read as a measured zero elsewhere in this module.
        """
        by_bucket: Dict[str, List[ShortRecord]] = defaultdict(list)
        for s in shorts:
            by_bucket[self._duration_bucket(s.content.duration_sec)].append(s)

        if not by_bucket or (len(by_bucket) == 1 and "unknown" in by_bucket):
            return ""

        order = ["≤15s", "16–30s", "31–60s", ">60s", "unknown"]
        lines = [
            "## ⏱️ Duration vs Performance",
            "",
            "| Duration | Shorts | Avg Engagement |",
            "|----------|--------|-----------------|",
        ]
        for bucket in order:
            group = by_bucket.get(bucket)
            if not group:
                continue
            lines.append(
                f"| {bucket} | {len(group)} | {_fmt_engagement_basis(_avg_engagement(group))} |"
            )

        return "\n".join(lines)

    def _platform_breakdown(self, shorts: List[ShortRecord]) -> str:
        """Per-platform metrics summary."""
        platform_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"views": 0, "likes": 0, "comments": 0, "shares": 0, "count": 0}
        )

        for s in shorts:
            for platform, metrics in s.metrics.items():
                pt = platform_totals[platform]
                # Unreported per-platform metrics don't contribute to this
                # cross-short aggregate — same rationale as
                # ShortRecord.total_views/total_engagement.
                pt["views"] += metrics.views or 0
                pt["likes"] += metrics.likes or 0
                pt["comments"] += metrics.comments or 0
                pt["shares"] += metrics.shares or 0
                pt["count"] += 1

        if not platform_totals:
            return ""

        lines = [
            "## 📊 Platform Breakdown",
            "",
            "| Platform | Shorts | Views | Likes | Comments | Shares | Engagement |",
            "|----------|--------|-------|-------|----------|--------|------------|",
        ]

        for platform, totals in sorted(platform_totals.items()):
            interactions = totals["likes"] + totals["comments"] + totals["shares"]
            eng = interactions / totals["views"] if totals["views"] > 0 else 0
            lines.append(
                f"| {platform.title()} | {int(totals['count'])} | "
                f"{int(totals['views']):,} | {int(totals['likes']):,} | "
                f"{int(totals['comments']):,} | {int(totals['shares']):,} | "
                f"{eng:.2%} |"
            )

        return "\n".join(lines)

    def _recommendations(
        self,
        shorts: List[ShortRecord],
        record: Optional[TrackingRecord] = None,
    ) -> str:
        """Generate actionable recommendations based on data patterns."""
        if not shorts:
            return "## Recommendations\n\nNot enough data yet.\n"

        recs: List[str] = []

        # ── Time recommendation ──────────────────────────────────────────
        by_hour: Dict[int, List[int]] = defaultdict(list)
        for s in shorts:
            h = s.posting_hour
            if h is not None:
                by_hour[h].append(s.total_views)

        if by_hour:
            best_hour = max(by_hour, key=lambda h: statistics.mean(by_hour[h]))
            worst_hour = min(by_hour, key=lambda h: statistics.mean(by_hour[h]))
            best_avg = statistics.mean(by_hour[best_hour])
            worst_avg = statistics.mean(by_hour[worst_hour])

            if best_avg > worst_avg * 1.3:  # >30% difference
                recs.append(
                    f"📅 **Post around {best_hour}:00** — shorts posted at this hour "
                    f"average {best_avg:,.0f} views vs {worst_avg:,.0f} at {worst_hour}:00"
                )

        # ── Weather recommendation ───────────────────────────────────────
        by_weather: Dict[str, List[int]] = defaultdict(list)
        for s in shorts:
            w = (s.content.weather or "").lower().strip()
            if w and w != "unknown":
                by_weather[w].append(s.total_views)

        if len(by_weather) >= 2:
            best_w = max(by_weather, key=lambda w: statistics.mean(by_weather[w]))
            best_w_avg = statistics.mean(by_weather[best_w])
            recs.append(
                f"🌤️ **{best_w.title()} weather performs best** — "
                f"avg {best_w_avg:,.0f} views. Prioritise these clips when scheduling."
            )

        # ── Hook recommendation ──────────────────────────────────────────
        by_hook: Dict[str, List[int]] = defaultdict(list)
        for s in shorts:
            hook = (s.content.hook or "").strip()
            if hook:
                by_hook[hook].append(s.total_views)

        if len(by_hook) >= 2:
            best_hook = max(by_hook, key=lambda h: statistics.mean(by_hook[h]))
            best_hook_avg = statistics.mean(by_hook[best_hook])
            recs.append(
                f'🪝 **Best hook: "{best_hook}"** — avg {best_hook_avg:,.0f} views. '
                f"Consider using similar phrasing."
            )

        # ── Engagement recommendation ────────────────────────────────────
        engagement_basis = _avg_engagement(shorts)
        if engagement_basis.value is not None:
            if engagement_basis.value < 0.02:
                recs.append(
                    "💬 **Low engagement rate** — consider adding a question in "
                    'the caption ("Can you guess?") or a stronger CTA.'
                )
            elif engagement_basis.value > 0.05:
                recs.append(
                    "🔥 **Strong engagement** — current strategy is working well. "
                    "Double down on high-engagement hooks."
                )

        # ── Volume recommendation ────────────────────────────────────────
        if len(shorts) < 10:
            recs.append(
                f"📊 **Only {len(shorts)} shorts tracked** — keep collecting data. "
                f"Patterns become reliable at 20+ shorts."
            )

        # ── Cross-reference with known formats ───────────────────────────
        guess_shorts = [s for s in shorts if "guess" in (s.content.title_format or "").lower()]
        other_shorts = [s for s in shorts if s not in guess_shorts and s.total_views > 0]
        if guess_shorts and other_shorts:
            guess_avg = statistics.mean([s.total_views for s in guess_shorts])
            other_avg = statistics.mean([s.total_views for s in other_shorts])
            if guess_avg > other_avg * 1.5:
                recs.append(
                    f'🎯 **"Guess the Location" format** averages {guess_avg:,.0f} views '
                    f"vs {other_avg:,.0f} for other formats ({guess_avg / other_avg:.1f}x better)"
                )

        if not recs:
            recs.append("✅ Not enough differentiation in the data yet — keep tracking!")

        lines = ["## 💡 Recommendations", ""]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    # ── Utility ──────────────────────────────────────────────────────────

    def get_insights_for_next_batch(
        self,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Machine-readable insights to feed into the pipeline.

        Returns a dict with recommended posting times, weather preferences,
        and hook suggestions — suitable for use by scheduler/social_generator.
        """
        if project:
            shorts = self.store.load(project).shorts_with_metrics
        else:
            shorts = [s for s in self.store.get_all_shorts() if s.metrics]

        if not shorts:
            return {
                "confidence": "low",
                "sample_size": 0,
                "recommendations": {},
            }

        insights: Dict[str, Any] = {
            "confidence": "high" if len(shorts) >= 20 else "medium" if len(shorts) >= 10 else "low",
            "sample_size": len(shorts),
            "recommendations": {},
        }

        # Best posting hours
        by_hour: Dict[int, List[int]] = defaultdict(list)
        for s in shorts:
            h = s.posting_hour
            if h is not None:
                by_hour[h].append(s.total_views)

        if by_hour:
            ranked = sorted(by_hour.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
            insights["recommendations"]["posting_hours"] = [
                {"hour": h, "avg_views": statistics.mean(v), "sample": len(v)}
                for h, v in ranked[:3]
            ]

        # Best weather
        by_weather: Dict[str, List[int]] = defaultdict(list)
        for s in shorts:
            w = (s.content.weather or "").lower().strip()
            if w and w != "unknown":
                by_weather[w].append(s.total_views)

        if by_weather:
            ranked = sorted(by_weather.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
            insights["recommendations"]["weather_preference"] = [
                {"weather": w, "avg_views": statistics.mean(v), "sample": len(v)} for w, v in ranked
            ]

        # Best hooks
        by_hook: Dict[str, List[int]] = defaultdict(list)
        for s in shorts:
            hook = (s.content.hook or "").strip()
            if hook:
                by_hook[hook].append(s.total_views)

        if by_hook:
            ranked = sorted(by_hook.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
            insights["recommendations"]["hooks"] = [
                {"hook": h, "avg_views": statistics.mean(v), "sample": len(v)}
                for h, v in ranked[:5]
            ]

        return insights
